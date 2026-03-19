"""
adapters/capture_transform_adapter.py — Capture-Transform ToolAdapter

Connects the capture-transform pipeline to MABE's orchestration layer.
Implements the ToolAdapter interface so the Orchestrator can route
capture-transform problems here automatically.

Recognizes problems where:
  - The desired outcome involves "transform", "convert", "feedstock",
    "mineralize", "precipitate", or "capture and convert"
  - The target is a gas (CO₂, SO₂, NH₃, N₂) or dissolved species
    (PO₄³⁻, NO₃⁻, F⁻, heavy metals) amenable to covalent transformation
  - The matrix is environmental (water, wastewater, air)

Maps pipeline SystemDesign output → CandidateResult for ranking
alongside candidates from other adapters (chelators, proteins, etc.).
"""

from __future__ import annotations

from adapters.base import ToolAdapter, Capability, ContributionAssessment
from core.problem import Problem
from core.candidate import (
    CandidateResult, PerformancePrediction, EvidenceProfile,
    AccessibilityProfile, ImmobilizationOption, ApplicationConnection,
)

from core.capture_transform_pipeline import (
    design_capture_transform_system,
    PipelineResult,
    SystemDesign,
)
from core.substrate_anchor import SubstrateType
from core.transform_enumerator import (
    TransformationProduct,
    ClickCompatibility,
)


# ═══════════════════════════════════════════════════════════════════════════
# Target formula mapping — Problem.target.formula → enumerator formula
# ═══════════════════════════════════════════════════════════════════════════

_FORMULA_MAP: dict[str, str] = {
    # CO2 family
    "CO2": "CO2", "CO₂": "CO2", "co2": "CO2",
    "HCO3(-)": "CO2", "CO3(2-)": "CO2",
    # Phosphate family
    "PO4(3-)": "PO4_3-", "HPO4(2-)": "PO4_3-", "H2PO4(-)": "PO4_3-",
    "PO₄³⁻": "PO4_3-", "phosphate": "PO4_3-",
    # Nitrogen family
    "NH3": "NH3", "NH₃": "NH3",
    "NH4(+)": "NH4+", "NH₄⁺": "NH4+",
    "NO3(-)": "NO3-", "NO₃⁻": "NO3-", "nitrate": "NO3-",
    "N2": "N2", "N₂": "N2",
    # Heavy metals
    "Pb(2+)": "Pb2+", "Pb²⁺": "Pb2+", "Pb2+": "Pb2+",
    "Cd(2+)": "Cd2+", "Cd²⁺": "Cd2+", "Cd2+": "Cd2+",
    "Hg(2+)": "Hg2+", "Hg²⁺": "Hg2+", "Hg2+": "Hg2+",
    "Cu(2+)": "Cu2+", "Cu²⁺": "Cu2+",
    "Zn(2+)": "Zn2+", "Zn²⁺": "Zn2+",
    # Anions
    "F(-)": "F-", "F⁻": "F-", "fluoride": "F-",
    "SO2": "SO2", "SO₂": "SO2",
    # Arsenic
    "H2AsO4(-)": "H2AsO4-", "AsO4(3-)": "H2AsO4-", "arsenate": "H2AsO4-",
    "As(V)": "H2AsO4-",
    # SeO3 — already handled by other adapters but could transform too
    "SeO3(2-)": "SeO3_2-",
}

# Identity-based mapping for decomposer targets that use .identity
_IDENTITY_MAP: dict[str, str] = {
    "carbon dioxide": "CO2",
    "phosphate": "PO4_3-",
    "ammonia": "NH3",
    "ammonium": "NH4+",
    "nitrate": "NO3-",
    "nitrogen": "N2",
    "lead": "Pb2+",
    "cadmium": "Cd2+",
    "mercury": "Hg2+",
    "fluoride": "F-",
    "sulfur dioxide": "SO2",
    "arsenic": "H2AsO4-",
    "arsenate": "H2AsO4-",
}

# Outcome keywords that signal capture-transform (vs capture-release)
_TRANSFORM_KEYWORDS = {
    "transform", "convert", "feedstock", "mineralize", "precipitate",
    "fertilizer", "mineral", "product", "solid", "crystal",
    "capture and convert", "capture and transform",
    "covalent", "react", "fix", "sequester",
}

# Targets that are inherently capture-transform (gases, dissolved toxins)
_INHERENT_TRANSFORM_TARGETS = {
    "CO2", "SO2", "NH3", "N2",  # gases
    "PO4_3-", "NO3-", "F-", "H2AsO4-",  # dissolved species for precipitation
    "Pb2+", "Cd2+", "Hg2+",  # heavy metals for sulfide precipitation
}


# ═══════════════════════════════════════════════════════════════════════════
# Matrix extraction
# ═══════════════════════════════════════════════════════════════════════════

def _extract_matrix_species(problem: Problem) -> dict[str, float]:
    """Extract matrix species concentrations from a Problem's Matrix."""
    species = {}
    if hasattr(problem.matrix, 'competing_species'):
        for cs in problem.matrix.competing_species:
            # Map formula format
            formula = cs.formula.replace("(", "").replace(")", "").replace(" ", "")
            # Common mappings
            formula_map = {
                "Ca2+": "Ca2+", "Ca(2+)": "Ca2+",
                "Mg2+": "Mg2+", "Mg(2+)": "Mg2+",
                "NH4+": "NH4+", "NH4(+)": "NH4+",
                "Fe3+": "Fe3+", "Fe(3+)": "Fe3+",
                "SO42-": "SO4_2-", "SO4(2-)": "SO4_2-",
            }
            clean = formula_map.get(formula, formula)
            species[clean] = cs.concentration_mm

    # Enrich from matrix description
    desc = getattr(problem.matrix, 'description', '').lower()
    if 'seawater' in desc or 'ocean' in desc:
        species.setdefault("Ca2+", 10.0)
        species.setdefault("Mg2+", 53.0)
    elif 'hard water' in desc:
        species.setdefault("Ca2+", 2.0)
        species.setdefault("Mg2+", 1.0)
    elif 'wastewater' in desc or 'agricultural' in desc or 'effluent' in desc:
        species.setdefault("NH4+", 5.0)
        species.setdefault("PO4_total", 0.5)
        species.setdefault("Ca2+", 2.0)
        species.setdefault("Mg2+", 1.0)
    elif 'mine' in desc or 'tailings' in desc:
        species.setdefault("Ca2+", 5.0)
        species.setdefault("Mg2+", 2.0)
        species.setdefault("Fe3+", 1.0)

    return species


def _resolve_target_formula(problem: Problem) -> str | None:
    """Resolve target formula from Problem to enumerator-compatible format."""
    # Try direct formula mapping
    formula = problem.target.formula
    if formula in _FORMULA_MAP:
        return _FORMULA_MAP[formula]

    # Try identity mapping
    identity = problem.target.identity.lower()
    if identity in _IDENTITY_MAP:
        return _IDENTITY_MAP[identity]

    # Try partial identity match
    for key, val in _IDENTITY_MAP.items():
        if key in identity:
            return val

    return None


def _is_transform_problem(problem: Problem) -> bool:
    """Detect if this problem is a capture-transform problem."""
    outcome = problem.desired_outcome.description.lower()
    query = problem.original_query.lower()

    # Check outcome keywords
    for kw in _TRANSFORM_KEYWORDS:
        if kw in outcome or kw in query:
            return True

    # Check if target is inherently transform-amenable
    resolved = _resolve_target_formula(problem)
    if resolved and resolved in _INHERENT_TRANSFORM_TARGETS:
        return True

    return False


def _estimate_scale(problem: Problem) -> str:
    """Estimate deployment scale from problem constraints."""
    scale_str = getattr(problem.constraints, 'scale', 'lab')
    query = problem.original_query.lower()

    if 'industrial' in query or 'plant' in query:
        return "industrial"
    elif 'field' in query or 'deployment' in query or 'real-world' in query:
        return "field"
    elif 'pilot' in query:
        return "pilot"
    elif 'diagnostic' in query or 'sensor' in query:
        return "diagnostic"
    else:
        return scale_str


# ═══════════════════════════════════════════════════════════════════════════
# SystemDesign → CandidateResult mapping
# ═══════════════════════════════════════════════════════════════════════════

def _design_to_candidate(design: SystemDesign, rank: int) -> CandidateResult:
    """Map a SystemDesign to a CandidateResult for the orchestrator."""
    ps = design.pathway_score
    product = ps.product

    # Performance prediction
    prob_success = min(0.95, ps.composite_score)
    confidence_str = "high" if ps.confidence >= 0.7 else "moderate" if ps.confidence >= 0.4 else "low"

    sensitive_to = []
    if ps.co_reactant_limited:
        sensitive_to.append("Co-reactant concentration in matrix")
    if ps.rate_class in ("slow", "very_slow"):
        sensitive_to.append("Kinetics — may need catalytic enhancement")

    performance = PerformancePrediction(
        probability_of_success=prob_success,
        confidence=confidence_str,
        confidence_reasoning=ps.confidence_basis,
        sensitive_to=sensitive_to,
        failure_modes=[ps.critical_risk] if ps.critical_risk else [],
        what_improves_odds=ps.advantages[:3],
        selectivity_threats=[l for l in ps.limitations if "selectiv" in l.lower()],
    )

    # Evidence
    evidence = EvidenceProfile(
        source_type="physics_and_literature",
        literature_references=[product.dg_source] if product.dg_source else [],
        computational_method="NIST thermochemical data + orthogonality scoring",
        what_would_validate=(
            f"Confirm {product.formula} formation on capture substrate "
            f"by XRD/FTIR/gravimetric analysis"
        ),
    )

    # Accessibility
    tether = design.tether_protocol
    equipment = ["Fume hood", "Analytical balance"]
    if tether and tether.activation.temperature_c > 50:
        equipment.append(f"Oven ({tether.activation.temperature_c}°C)")
    if any("plasma" in s.lower() for s in (tether.activation.steps if tether else [])):
        equipment.append("Plasma cleaner")

    regen = ps.regeneration
    cycles = regen.estimated_cycles if regen else 0

    accessibility = AccessibilityProfile(
        estimated_cost="$50-500 per batch (lab scale)" if ps.deployment_scale == "lab" else "$10-100/kg (field scale)",
        equipment_required=equipment,
        community_lab_feasible=(ps.deployment_scale in ("lab", "diagnostic")),
        reusability_cycles=cycles if cycles > 1 else None,
        waste_generated="Product is feedstock — minimal waste" if product.harvestable else "Spent substrate",
        end_of_life="Product harvested as " + product.feedstock_value if product.feedstock_value else "Disposal",
    )

    # Immobilization options
    immob = []
    if tether:
        immob.append(ImmobilizationOption(
            substrate=tether.substrate.name,
            attachment_chemistry=tether.click_chemistry.value,
            click_handle=tether.handle.handle_installed,
            effect_on_binding="Tether adds PEG spacer — minimal steric impact on capture site",
            notes=f"Cu-free: {tether.cu_safe}",
        ))

    # Cross-domain applications
    other_apps = []
    if product.feedstock_value:
        other_apps.append(ApplicationConnection(
            domain="industrial feedstock",
            description=product.feedstock_value,
            what_would_change="Product harvesting and purification protocol",
            confidence="high" if ps.confidence >= 0.7 else "moderate",
        ))
    if design.cascade_spec and design.scaffold_recommendation:
        other_apps.append(ApplicationConnection(
            domain="materials science",
            description=f"Cascade scaffold ({design.scaffold_recommendation.scaffold.name}) "
                        f"design transferable to other capture-transform targets",
            what_would_change="Swap capture modules for different targets",
            confidence="moderate",
        ))

    # Structure description
    site_name = product.capture_sites[0].name if product.capture_sites else "capture site"
    structure_desc = (
        f"Capture-transform element: {site_name}\n"
        f"Reaction: {product.target_formula} → {product.formula}\n"
        f"ΔG_total = {ps.dg_total_kj:+.1f} kJ/mol\n"
        f"Turnover: {product.turnover.value}\n"
        f"Product phase: {product.product_phase.value}"
    )
    if design.scaffold_recommendation and design.scaffold_recommendation.feasible:
        structure_desc += (
            f"\nScaffold: {design.scaffold_recommendation.scaffold.name}"
            f" (confinement: {design.scaffold_recommendation.confinement_factor:.0f}×)"
        )
    if tether:
        structure_desc += f"\nSubstrate: {tether.substrate.name} via {tether.click_chemistry.value}"

    # Name
    tier = "GOLD" if ps.is_gold else "SILVER" if ps.orthogonality_score >= 0.4 else "BRONZE"
    name = f"[{tier}] {ps.name}"

    return CandidateResult(
        rank=rank,
        name=name,
        description=f"Capture-transform pathway: {product.target_formula} → {product.formula}. "
                    f"Orthogonality: {ps.orthogonality_score:.2f}. "
                    f"System score: {design.system_score:.3f}.",
        modality="capture_transform",
        source_tool="capture_transform_pipeline",
        structure_description=structure_desc,
        performance=performance,
        evidence=evidence,
        accessibility=accessibility,
        immobilization_options=immob,
        other_applications=other_apps,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ToolAdapter implementation
# ═══════════════════════════════════════════════════════════════════════════

class CaptureTransformAdapter(ToolAdapter):
    """
    Capture-transform adapter for MABE orchestration.

    Recognizes problems involving irreversible covalent transformation of
    captured targets into harvestable feedstock. Routes to the full
    capture-transform pipeline (enumerator → scorer → substrate → cascade).
    """

    @property
    def name(self) -> str:
        return "capture_transform"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> list[Capability]:
        return [
            Capability(
                description=(
                    "Design capture-transform systems that selectively bind a target "
                    "from air or water and drive irreversible covalent conversion into "
                    "harvestable solid feedstock. Includes orthogonality scoring (no "
                    "external reagents/energy), substrate tethering via click chemistry, "
                    "and 3D scaffold cascade design."
                ),
                target_types=[
                    "gas (CO₂, SO₂, NH₃, N₂)",
                    "dissolved anion (PO₄³⁻, NO₃⁻, F⁻, AsO₄³⁻)",
                    "heavy metal ion (Pb²⁺, Cd²⁺, Hg²⁺, Cu²⁺, Zn²⁺)",
                ],
                interaction_types=[
                    "covalent transformation",
                    "mineral precipitation",
                    "acid-base capture",
                    "photocatalytic reduction",
                    "metal sulfide precipitation",
                ],
                output_types=[
                    "transformation product enumeration",
                    "orthogonality-scored pathways",
                    "substrate tethering protocols",
                    "cascade scaffold recommendations",
                    "system-level ranked designs",
                ],
            ),
        ]

    def is_available(self) -> bool:
        return True  # no external dependencies

    def assess_contribution(self, problem: Problem) -> ContributionAssessment:
        """Assess whether this problem is a capture-transform problem."""

        resolved_formula = _resolve_target_formula(problem)
        is_transform = _is_transform_problem(problem)

        # Strong match: known target + transform outcome
        if resolved_formula and is_transform:
            return ContributionAssessment(
                can_contribute=True,
                relevance=0.95,
                what_it_would_do=(
                    f"Design capture-transform system for {problem.target.identity}: "
                    f"enumerate covalent products, score orthogonality, generate "
                    f"substrate tethering protocol, recommend cascade scaffold"
                ),
                what_part_of_problem="capture-transform system design (covalent conversion to feedstock)",
                estimated_compute_time="<1 second",
                limitations=[
                    "Orthogonal pathways only (no external reagents or active energy input)",
                    "Thermodynamic data from NIST — activation barriers estimated",
                    "Substrate protocols are lab-verified recipes, not optimized for specific system",
                ],
            )

        # Moderate match: known target, no explicit transform keyword
        # (many targets are inherently transform-amenable)
        if resolved_formula and resolved_formula in _INHERENT_TRANSFORM_TARGETS:
            return ContributionAssessment(
                can_contribute=True,
                relevance=0.6,
                what_it_would_do=(
                    f"Evaluate capture-transform pathways for {problem.target.identity} — "
                    f"target is amenable to covalent transformation even though "
                    f"outcome doesn't explicitly request it"
                ),
                what_part_of_problem="alternative approach: capture-transform (vs capture-release)",
                estimated_compute_time="<1 second",
                limitations=[
                    "This is an alternative to capture-release — may not be what user intended",
                ],
            )

        # No match
        return ContributionAssessment(
            can_contribute=False,
            relevance=0.0,
            what_it_would_do="N/A",
            what_part_of_problem="N/A",
            limitations=["Target not in capture-transform database"],
        )

    def generate_candidates(self, problem: Problem) -> list[CandidateResult]:
        """Generate capture-transform candidates from the full pipeline."""

        resolved_formula = _resolve_target_formula(problem)
        if not resolved_formula:
            return []

        matrix_species = _extract_matrix_species(problem)
        scale = _estimate_scale(problem)

        # Run the full pipeline
        result = design_capture_transform_system(
            target_formula=resolved_formula,
            matrix_species=matrix_species,
            substrate_type=SubstrateType.SILICA_BEADS,  # default
            dg_bind_kj=-20.0,  # estimated default
            temperature_c=getattr(problem.matrix, 'temperature_c', 25.0),
            target_scale=scale,
            max_designs=6,
        )

        # Map to CandidateResult format
        candidates = []
        for i, design in enumerate(result.designs):
            if design.pathway_score.orthogonality_score > 0:  # only viable pathways
                candidate = _design_to_candidate(design, rank=i + 1)
                candidates.append(candidate)

        return candidates
