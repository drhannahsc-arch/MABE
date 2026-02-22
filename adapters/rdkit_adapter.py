"""
adapters/rdkit_adapter.py — RDKit adapter for MABE.

First real adapter. Generates and evaluates chelator candidates using
actual molecular structures, computed properties, and physics-based
scoring against the target species.

What it does:
1. Searches the chelator library for templates matching the target's physics
2. Computes molecular properties (MW, LogP, TPSA, donors, acceptors, rotatable bonds)
3. Scores candidates based on HSAB match, denticity vs coordination number,
   known stability constants, and matrix compatibility
4. Returns real CandidateResults with actual SMILES and computed properties
"""

from __future__ import annotations

from adapters.base import ToolAdapter, Capability, ContributionAssessment
from core.problem import Problem
from core.candidate import (
    CandidateResult, PerformancePrediction, EvidenceProfile,
    AccessibilityProfile, ImmobilizationOption, ApplicationConnection,
)
from knowledge.chelator_library import CHELATOR_TEMPLATES, ELEMENT_TO_TARGETS

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def _compute_mol_properties(smiles: str) -> dict:
    """Compute molecular properties from SMILES using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Could not parse SMILES: {smiles}"}

    return {
        "molecular_weight": round(Descriptors.MolWt(mol), 1),
        "logP": round(Descriptors.MolLogP(mol), 2),
        "tpsa": round(Descriptors.TPSA(mol), 1),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "num_atoms": mol.GetNumHeavyAtoms(),
        "num_rings": rdMolDescriptors.CalcNumRings(mol),
        "canonical_smiles": Chem.MolToSmiles(mol),
    }


def _hsab_match_score(chelator_donor_type: str, target_hardness: str) -> float:
    """Score how well donor type matches target HSAB classification."""
    if target_hardness is None or target_hardness == "unknown":
        return 0.5  # unknown — no penalty, no bonus

    match_matrix = {
        ("hard", "hard"): 1.0,
        ("hard", "borderline"): 0.6,
        ("hard", "soft"): 0.2,
        ("soft", "soft"): 1.0,
        ("soft", "borderline"): 0.6,
        ("soft", "hard"): 0.2,
        ("borderline", "borderline"): 0.8,
        ("borderline", "hard"): 0.6,
        ("borderline", "soft"): 0.6,
    }
    return match_matrix.get((chelator_donor_type, target_hardness), 0.5)


def _denticity_match_score(chelator_denticity: int, target_coord_number: int) -> float:
    """Score how well chelator denticity matches target coordination preference."""
    if target_coord_number is None or target_coord_number == 0:
        return 0.5
    ratio = chelator_denticity / target_coord_number
    if 0.8 <= ratio <= 1.0:
        return 1.0  # denticity matches or slightly under (leaves room for solvent)
    elif 0.5 <= ratio <= 1.2:
        return 0.7
    else:
        return 0.3


def _ph_compatibility_score(chelator_ph_range: tuple, matrix_ph: float) -> float:
    """Score pH compatibility."""
    if matrix_ph is None:
        return 0.7  # assume moderate compatibility
    low, high = chelator_ph_range
    if low <= matrix_ph <= high:
        return 1.0
    elif abs(matrix_ph - low) <= 1.0 or abs(matrix_ph - high) <= 1.0:
        return 0.5  # marginal
    else:
        return 0.1  # outside range


def _find_target_element(target_identity: str) -> str:
    """Map target identity to element symbol for log_k lookup."""
    identity = target_identity.lower()
    for element, names in ELEMENT_TO_TARGETS.items():
        if identity in names:
            return element
    return ""


def _find_element_in_logk(element: str, log_k_range: dict) -> tuple:
    """Find element in log_k dict, trying aliases (Fe -> Fe3, etc)."""
    if element in log_k_range:
        return element, log_k_range[element]
    # Try with common oxidation state suffixes
    for alias in [element + "3", element + "2", element.rstrip("0123456789")]:
        if alias in log_k_range:
            return alias, log_k_range[alias]
    return element, None


def _stability_score(log_k_range: dict, target_element: str) -> tuple[float, str]:
    """Score based on known stability constants. Returns (score, log_k_str)."""
    if not target_element:
        return 0.3, "no data"
    _, log_k = _find_element_in_logk(target_element, log_k_range)
    if log_k is None:
        return 0.3, "no data"
    if log_k >= 20:
        return 1.0, f"log K = {log_k}"
    elif log_k >= 15:
        return 0.8, f"log K = {log_k}"
    elif log_k >= 10:
        return 0.6, f"log K = {log_k}"
    elif log_k >= 5:
        return 0.4, f"log K = {log_k}"
    else:
        return 0.2, f"log K = {log_k}"


def _selectivity_analysis(log_k_range: dict, target_element: str,
                           competing_species: list) -> list[str]:
    """Identify selectivity threats from competing species."""
    threats = []
    if not target_element:
        threats.append("No stability constant data - selectivity unknown")
        return threats

    target_key, target_log_k = _find_element_in_logk(target_element, log_k_range)
    if target_log_k is None:
        threats.append(f"No stability constant for {target_element} - selectivity unknown")
        return threats

    for comp in competing_species:
        comp_element = _find_target_element(comp.identity)
        if comp_element:
            comp_key, comp_log_k = _find_element_in_logk(comp_element, log_k_range)
            if comp_log_k is not None:
                diff = target_log_k - comp_log_k
                if diff < 1.0:
                    threats.append(
                        f"{comp.identity} ({comp.formula}) competes strongly - "
                        f"log K difference only {diff:.1f} "
                        f"(target={target_log_k}, competitor={comp_log_k})"
                    )
                elif diff < 3.0:
                    threats.append(
                        f"{comp.identity} ({comp.formula}) may interfere at high concentrations - "
                        f"log K difference {diff:.1f}"
                    )
    return threats


class RDKitAdapter(ToolAdapter):
    """
    Real chemistry adapter. Evaluates chelator candidates using RDKit
    for molecular property computation and physics-based scoring.
    """

    @property
    def name(self) -> str:
        return "rdkit_chelator"

    @property
    def version(self) -> str:
        return "0.2.0"

    @property
    def capabilities(self) -> list[Capability]:
        return [
            Capability(
                description="Generate and evaluate small-molecule chelators with computed molecular properties",
                target_types=["metal_ion"],
                interaction_types=["coordination", "chelation"],
                output_types=["molecular_structure", "binding_estimate", "molecular_properties"],
            ),
        ]

    def is_available(self) -> bool:
        return RDKIT_AVAILABLE

    def assess_contribution(self, problem: Problem) -> ContributionAssessment:
        if not RDKIT_AVAILABLE:
            return ContributionAssessment(
                can_contribute=False, relevance=0.0,
                what_it_would_do="RDKit not installed",
                what_part_of_problem="none",
                limitations=["pip install rdkit"],
            )

        # Can contribute if target looks like a metal ion
        charge = problem.target.charge
        has_donors = bool(problem.target.electronic.donor_atoms)
        has_coord = problem.target.geometry and "unknown" not in problem.target.geometry.lower()

        if abs(charge) > 0 or has_donors or has_coord:
            return ContributionAssessment(
                can_contribute=True,
                relevance=0.8,
                what_it_would_do=(
                    f"Evaluate chelator templates against {problem.target.identity} "
                    f"using HSAB matching, coordination geometry, stability constants, "
                    f"and computed molecular properties"
                ),
                what_part_of_problem="molecular recognition design (chelator modality)",
                estimated_compute_time="seconds",
                limitations=[
                    "Limited to small-molecule chelators (not proteins or DNA)",
                    "Stability constants from literature — may not cover all targets",
                    "Does not model dynamics or solvation explicitly",
                ],
            )
        else:
            return ContributionAssessment(
                can_contribute=False, relevance=0.1,
                what_it_would_do="Target does not appear to be a metal ion",
                what_part_of_problem="none",
            )

    def generate_candidates(self, problem: Problem) -> list[CandidateResult]:
        if not RDKIT_AVAILABLE:
            return []

        target = problem.target
        matrix = problem.matrix
        target_element = _find_target_element(target.identity)
        target_hsab = target.electronic.hardness_softness
        target_coord = target.hydration.coordination_number_water

        candidates = []

        for template in CHELATOR_TEMPLATES:
            # Score this template against the target
            hsab_score = _hsab_match_score(template["donor_type"], target_hsab)
            dent_score = _denticity_match_score(template["denticity"], target_coord)
            ph_score = _ph_compatibility_score(template["ph_optimal"], matrix.ph)
            stab_score, stab_str = _stability_score(template["log_k_range"], target_element)

            # Combined score
            overall = (hsab_score * 0.30 + stab_score * 0.30 +
                       ph_score * 0.20 + dent_score * 0.20)

            # Skip very poor matches
            if overall < 0.25:
                continue

            # Compute real molecular properties
            props = _compute_mol_properties(template["smiles"])
            if "error" in props:
                continue

            # Selectivity analysis
            threats = _selectivity_analysis(
                template["log_k_range"], target_element,
                matrix.competing_species
            )

            # Build failure modes
            failure_modes = []
            if ph_score < 0.5:
                failure_modes.append(
                    f"Matrix pH ({matrix.ph}) is marginal for this chelator "
                    f"(optimal range: {template['ph_optimal'][0]}-{template['ph_optimal'][1]})"
                )
            if template["donor_type"] == "soft":
                failure_modes.append("Thiol/soft donors vulnerable to oxidation in aerobic conditions")
            if template["denticity"] < 4:
                failure_modes.append("Low denticity — complex may not be kinetically stable in flow")

            # Build improvement suggestions
            improvements = []
            if ph_score < 1.0 and matrix.ph is not None:
                optimal_ph = (template["ph_optimal"][0] + template["ph_optimal"][1]) / 2
                improvements.append(f"Buffer to pH {optimal_ph:.1f} for optimal chelation")
            if threats:
                improvements.append("Pre-treatment to reduce competing ion concentrations")
            if template["donor_type"] == "soft" and matrix.redox_potential_mv and matrix.redox_potential_mv > 200:
                improvements.append("Add reducing agent or deoxygenate to protect thiol donors")

            # Confidence
            if stab_str != "no data" and hsab_score >= 0.6:
                confidence = "moderate"
                confidence_reason = (
                    f"Stability constant known ({stab_str}), HSAB match is "
                    f"{'good' if hsab_score >= 0.8 else 'reasonable'}. "
                    f"Matrix compatibility {'confirmed' if ph_score >= 0.8 else 'uncertain'}."
                )
            elif stab_str != "no data":
                confidence = "low"
                confidence_reason = (
                    f"Stability constant known ({stab_str}) but HSAB match is poor "
                    f"({template['donor_type']} donors vs {target_hsab} target)."
                )
            else:
                confidence = "low"
                confidence_reason = (
                    f"No stability constant data for {target.identity}. "
                    f"Prediction based on HSAB matching and coordination geometry only."
                )

            # Evidence
            if stab_str != "no data":
                evidence_type = "hybrid"
                lit_refs = [f"Stability constants: Martell & Smith Critical Stability Constants ({stab_str})"]
            else:
                evidence_type = "computational_prediction"
                lit_refs = []

            # Structure description with real properties
            struct_desc = (
                f"{props['canonical_smiles']} | "
                f"MW={props['molecular_weight']}, LogP={props['logP']}, "
                f"TPSA={props['tpsa']}, {props['hba']}xHBA, {props['hbd']}xHBD, "
                f"{template['denticity']}-dentate {'/'.join(template['donor_atoms'])} donors"
            )

            # Cross-domain connections
            other_apps = []
            if template.get("notes") and "FDA" in template["notes"]:
                other_apps.append(ApplicationConnection(
                    domain="therapeutic",
                    description=f"This chelator is FDA-approved — direct clinical translation path exists",
                    what_would_change="Formulation and dosing optimization for target indication",
                    confidence="strong",
                ))
            if overall > 0.6 and target_hsab in ("soft", "borderline"):
                other_apps.append(ApplicationConnection(
                    domain="diagnostic",
                    description=f"High-affinity {target.identity} chelator could serve as sensor recognition element",
                    what_would_change="Conjugate to electrochemical or fluorescent reporter",
                    confidence="plausible",
                ))

            candidates.append(CandidateResult(
                rank=0,
                name=f"{template['name']} for {target.identity}",
                description=(
                    f"{template['name']} — a {template['denticity']}-dentate chelator with "
                    f"{'/'.join(template['donor_atoms'])} donor atoms ({template['donor_type']} donors). "
                    f"{template.get('notes', '')} "
                    f"Computed properties: MW {props['molecular_weight']}, "
                    f"LogP {props['logP']}, TPSA {props['tpsa']}."
                ),
                modality="chelator",
                source_tool="rdkit_chelator",
                structure_description=struct_desc,
                performance=PerformancePrediction(
                    probability_of_success=round(min(overall, 0.95), 2),
                    confidence=confidence,
                    confidence_reasoning=confidence_reason,
                    sensitive_to=[
                        f"pH — optimal range {template['ph_optimal'][0]}-{template['ph_optimal'][1]}",
                    ],
                    failure_modes=failure_modes if failure_modes else ["No major failure modes identified at this stage"],
                    what_improves_odds=improvements if improvements else ["Conditions appear favorable"],
                    selectivity_threats=threats if threats else ["No major selectivity threats identified from known constants"],
                ),
                evidence=EvidenceProfile(
                    source_type=evidence_type,
                    literature_references=lit_refs,
                    computational_method="RDKit molecular property computation + HSAB/stability constant scoring",
                    what_would_validate=f"ITC or SPR binding assay: {template['name']} vs {target.identity} and top 3 competitors",
                ),
                accessibility=AccessibilityProfile(
                    estimated_cost=template["cost_per_gram"],
                    equipment_required=["analytical balance", "pH meter"],
                    community_lab_feasible=("commodity" in template["accessibility"] or "any lab" in template["accessibility"]),
                    reusability_cycles=100 if template["donor_type"] == "hard" else 30,
                    waste_generated="Eluent with captured metal — recover by precipitation or electrodeposition",
                    end_of_life="Organic compound — biodegradable" if props["molecular_weight"] < 500 else "May require disposal protocol",
                ),
                immobilization_options=[
                    ImmobilizationOption(
                        substrate="nylon netting",
                        attachment_chemistry="Functionalize with amine linker, NHS coupling to nylon",
                        click_handle="carboxylate or amine group on chelator",
                        effect_on_binding=f"Reduces 1-2 coordination sites if carboxylate used for attachment. "
                                         f"Use linker arm (C6+) to minimize steric interference.",
                    ),
                    ImmobilizationOption(
                        substrate="silica beads (column)",
                        attachment_chemistry="Silanization + amide coupling",
                        click_handle="carboxylate group",
                        effect_on_binding="Minimal with adequate linker length. Well-established protocol.",
                    ),
                ],
                other_applications=other_apps,
            ))

        # Sort by overall score (stored in probability_of_success)
        candidates.sort(key=lambda c: c.performance.probability_of_success, reverse=True)

        # Take top 8 to avoid overwhelming
        candidates = candidates[:8]

        return candidates
