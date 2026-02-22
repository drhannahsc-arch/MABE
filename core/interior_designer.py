"""
core/interior_designer.py - Designs binding pocket interiors.

Given a target, a structural constraint, and available recognition chemistries,
designs the optimal interior arrangement:

Level 1 (simple): Best single recognition element, free.
Level 2 (composite): N copies of one recognition element in a cage.
Level 3 (tertiary): Multiple DIFFERENT elements arranged to create a pocket.

The tertiary design is the key innovation. It asks:
- What combination of donor types best matches this target?
- How many of each, at what spacing?
- Does mixing hard + soft donors create selectivity neither has alone?
- Does the spacing match the target's coordination geometry?
- Does the pore size create kinetic trapping?
"""

from __future__ import annotations
import math
from core.assembly import (
    RecognitionChemistry, InteriorDesign, InteriorSite,
    StructuralConstraint,
)
from core.problem import Problem
from core.candidate import CandidateResult


def _estimate_avidity(n_sites: int, kd_monovalent_um: float,
                       spacing_nm: float, target_radius_nm: float) -> float:
    """
    Estimate avidity enhancement from multivalent binding.

    Crude but physically grounded model:
    - If sites are within 2x target coordination shell, cooperative
    - Avidity scales roughly as N^(0.5-0.8) for flexible linkers
    - Rigid scaffolds (DNA origami) get better scaling than flexible (polymer)
    """
    if n_sites <= 1:
        return 1.0
    if spacing_nm is None or spacing_nm == 0:
        return float(n_sites) ** 0.5  # flexible, conservative

    # Target can "reach" multiple sites if spacing < 2 * hydrated radius
    reach = target_radius_nm * 2
    if spacing_nm <= reach * 3:
        # Sites close enough for cooperative rebinding
        # Rigid scaffold = better scaling
        return float(n_sites) ** 0.7
    else:
        # Sites too far apart, independent binding, additive not cooperative
        return float(n_sites) ** 0.5


def _pick_complementary_recognition(primary: RecognitionChemistry,
                                     all_candidates: list[CandidateResult],
                                     target_identity: str) -> RecognitionChemistry | None:
    """
    For tertiary design: find a complementary recognition element.

    Complementary means different donor type that addresses a weakness
    of the primary. E.g.:
    - Primary is hard O/N donor → complement with soft S donor for
      borderline metal → mixed pocket has selectivity neither has alone
    - Primary is chelator → complement with DNAzyme for different
      binding geometry
    """
    primary_type = primary.type
    primary_donors = set(primary.donor_atoms)

    best = None
    best_score = 0.0

    for c in all_candidates:
        if c.source_tool == "dummy":
            continue
        if c.name == primary.source_candidate_name:
            continue
        if target_identity.lower() not in c.name.lower():
            continue

        # Score complementarity
        score = 0.0
        modality = c.modality.lower()

        # Different modality type = good
        rec_type = modality.replace("_chelator", "").replace("dna_", "")
        if rec_type != primary_type:
            score += 0.3

        # Different donor atoms = excellent for mixed pocket
        desc = c.description.lower()
        candidate_donors = set()
        if "thiol" in desc or "/s" in c.structure_description.lower() or "cys" in desc:
            candidate_donors.add("S")
        if "/n" in c.structure_description.lower() or "his" in desc or "amine" in desc:
            candidate_donors.add("N")
        if "/o" in c.structure_description.lower() or "carbox" in desc:
            candidate_donors.add("O")

        new_donors = candidate_donors - primary_donors
        if new_donors:
            score += 0.4  # brings new donor types

        # Reasonable probability
        score += c.performance.probability_of_success * 0.3

        if score > best_score:
            best_score = score
            # Extract recognition from candidate
            best = RecognitionChemistry(
                name=c.name,
                type=rec_type,
                donor_atoms=list(candidate_donors) if candidate_donors else ["unknown"],
                donor_type="soft" if "S" in candidate_donors else ("hard" if "O" in candidate_donors else "borderline"),
                structure=c.structure_description.split("|")[0].strip(),
                kd_um=None,
                source_tool=c.source_tool,
                source_candidate_name=c.name,
                notes=c.description[:150],
            )

    # Only return if meaningfully complementary
    if best_score >= 0.5:
        return best
    return None


def design_interior(candidate: CandidateResult,
                     structure: StructuralConstraint,
                     problem: Problem,
                     all_candidates: list[CandidateResult]) -> InteriorDesign:
    """
    Design the interior arrangement for a binder assembly.

    Returns an InteriorDesign describing what goes where and why.
    """
    recognition = _extract_recognition_from_candidate(candidate)
    target = problem.target
    target_radius_nm = 0.3
    if target.size and target.size.hydrated_radius_angstrom:
        target_radius_nm = target.size.hydrated_radius_angstrom / 10.0

    # ── LEVEL 1: Simple (no structure) ────────────────────────────
    if structure.type == "none":
        return InteriorDesign(
            sites=[InteriorSite(
                recognition=recognition,
                copies=1,
                position_description="free in solution",
                attachment_chemistry="N/A",
            )],
            design_level="simple",
            total_binding_sites=1,
            unique_recognition_types=1,
            avidity_factor=1.0,
            design_rationale="Single recognition element, no structural constraint. Simplest deployment.",
        )

    # ── Determine how many sites fit ──────────────────────────────
    max_sites = structure.max_interior_sites
    spacing = structure.recognition_spacing_nm

    # ── LEVEL 2: Composite (N copies of same element) ─────────────
    # Use for: strong recognition where avidity just needs more copies
    n_copies = min(max_sites, 12)  # practical limit for staple modifications

    kd = recognition.kd_um or 10.0
    avidity = _estimate_avidity(n_copies, kd, spacing, target_radius_nm)

    composite_design = InteriorDesign(
        sites=[InteriorSite(
            recognition=recognition,
            copies=n_copies,
            position_description=f"distributed across {structure.geometry} interior, ~{spacing} nm spacing" if spacing else "distributed in interior",
            attachment_chemistry=_pick_attachment(recognition.type, structure.type),
        )],
        design_level="composite",
        total_binding_sites=n_copies,
        unique_recognition_types=1,
        avidity_factor=round(avidity, 1),
        geometric_match=_describe_geometric_match(recognition, target, spacing),
        kinetic_trapping=_describe_kinetic_trapping(structure, target_radius_nm),
        design_rationale=(
            f"{n_copies} copies of {recognition.name} at ~{spacing} nm spacing. "
            f"Avidity enhancement: {avidity:.0f}x over monovalent. "
            f"Target enters through {structure.pore_size_nm} nm pore, encounters dense binding field."
        ),
    )

    # ── LEVEL 3: Tertiary (mixed recognition = designed pocket) ───
    # Try to find complementary recognition for a mixed pocket
    complement = _pick_complementary_recognition(
        recognition, all_candidates, target.identity
    )

    if complement is not None:
        # Split sites between primary and complement
        primary_n = max(1, int(n_copies * 0.6))
        complement_n = max(1, min(n_copies - primary_n, int(n_copies * 0.4)))

        # Combined donors
        all_donors = set(recognition.donor_atoms) | set(complement.donor_atoms)

        # Mixed avidity — conservative estimate, but cooperativity bonus
        mixed_avidity = _estimate_avidity(primary_n + complement_n, kd, spacing, target_radius_nm)
        cooperativity_bonus = 1.3 if len(all_donors) > len(set(recognition.donor_atoms)) else 1.0
        mixed_avidity *= cooperativity_bonus

        cooperativity_note = _describe_cooperativity(
            recognition, complement, target, all_donors
        )

        tertiary_design = InteriorDesign(
            sites=[
                InteriorSite(
                    recognition=recognition,
                    copies=primary_n,
                    position_description=f"primary sites, {structure.geometry} interior",
                    attachment_chemistry=_pick_attachment(recognition.type, structure.type),
                ),
                InteriorSite(
                    recognition=complement,
                    copies=complement_n,
                    position_description=f"complementary sites, interspersed with primary",
                    attachment_chemistry=_pick_attachment(complement.type, structure.type),
                ),
            ],
            design_level="tertiary",
            total_binding_sites=primary_n + complement_n,
            unique_recognition_types=2,
            avidity_factor=round(mixed_avidity, 1),
            cooperativity_note=cooperativity_note,
            geometric_match=_describe_geometric_match(recognition, target, spacing),
            kinetic_trapping=_describe_kinetic_trapping(structure, target_radius_nm),
            design_rationale=(
                f"Mixed pocket: {primary_n}x {recognition.name} + {complement_n}x {complement.name}. "
                f"Combined donors: {'/'.join(sorted(all_donors))}. "
                f"{cooperativity_note} "
                f"Effective avidity: {mixed_avidity:.0f}x over monovalent."
            ),
        )

        return tertiary_design

    # If no good complement, return composite
    return composite_design


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def _extract_recognition_from_candidate(candidate: CandidateResult) -> RecognitionChemistry:
    """Extract recognition chemistry from a flat candidate."""
    modality = candidate.modality.lower()
    donor_atoms = []
    donor_type = "unknown"
    desc_lower = candidate.description.lower()

    if "thiol" in desc_lower or "/s" in candidate.structure_description.lower() or "cys" in desc_lower:
        donor_atoms.append("S")
        donor_type = "soft"
    if "/n" in candidate.structure_description.lower() or "his" in desc_lower or "amine" in desc_lower or "nitrogen" in desc_lower:
        donor_atoms.append("N")
    if "/o" in candidate.structure_description.lower() or "carbox" in desc_lower or "hydroxam" in desc_lower:
        donor_atoms.append("O")
    if "hard" in desc_lower:
        donor_type = "hard"
    elif "borderline" in desc_lower:
        donor_type = "borderline"
    elif "soft" in desc_lower:
        donor_type = "soft"
    if not donor_atoms:
        donor_atoms = ["O", "N"]  # default for most chelators

    kd = None
    if "Kd=" in candidate.structure_description:
        try:
            kd_str = candidate.structure_description.split("Kd=")[1].split(" ")[0].split(",")[0]
            kd = float(kd_str)
        except (ValueError, IndexError):
            pass

    return RecognitionChemistry(
        name=candidate.name,
        type=modality.replace("_chelator", "").replace("dna_", ""),
        donor_atoms=donor_atoms,
        donor_type=donor_type,
        structure=candidate.structure_description.split("|")[0].strip(),
        kd_um=kd,
        source_tool=candidate.source_tool,
        source_candidate_name=candidate.name,
        notes=candidate.description[:200],
    )


def _pick_attachment(recognition_type: str, structure_type: str) -> str:
    """Pick appropriate attachment chemistry for recognition on structure."""
    if structure_type in ("dna_origami_cage",):
        if recognition_type in ("dnazyme", "aptamer", "motif"):
            return "Staple overhang extension (hybridization)"
        elif recognition_type in ("chelator",):
            return "DTPA/EDTA-modified staple (custom oligo synthesis)"
        elif recognition_type in ("peptide",):
            return "DBCO-azide click to azide-modified staple overhang"
        else:
            return "NHS-amine coupling to amine-modified staple terminus"
    elif structure_type == "mof":
        return "Post-synthetic modification of linker functional groups"
    elif structure_type == "protein_cage":
        return "Genetic fusion or SpyCatcher-SpyTag conjugation"
    elif structure_type == "dendrimer":
        return "NHS coupling to surface amine groups"
    elif structure_type in ("mesoporous_silica", "silica_np"):
        return "Organosilane grafting (APTES, MPTMS, or custom silane)"
    elif structure_type == "cof":
        return "Post-synthetic modification of COF linker functional groups"
    elif structure_type in ("carbon_nanotube", "graphene_oxide"):
        return "Carbodiimide coupling to surface -COOH groups"
    elif structure_type == "coordination_cage":
        return "Ligand functionalization with pendant binding group"
    else:
        return "Standard bioconjugation"


def _describe_geometric_match(recognition: RecognitionChemistry,
                                target, spacing) -> str:
    """Describe how recognition spacing matches target coordination."""
    if spacing is None:
        return ""

    coord_num = None
    if target.hydration and target.hydration.coordination_number_water:
        coord_num = target.hydration.coordination_number_water

    target_radius = 0.3
    if target.size and target.size.ionic_radius_angstrom:
        target_radius = target.size.ionic_radius_angstrom

    if coord_num:
        return (
            f"Target {target.identity} has coordination number {coord_num} "
            f"(ionic radius {target_radius} A). "
            f"Interior sites at {spacing} nm spacing — target can bridge "
            f"between adjacent sites for multidentate coordination."
        )
    return f"Interior sites at {spacing} nm spacing."


def _describe_kinetic_trapping(structure: StructuralConstraint,
                                 target_radius_nm: float) -> str:
    """Describe how pore + interior creates kinetic advantage."""
    if structure.pore_size_nm is None or structure.type == "none":
        return ""

    pore = structure.pore_size_nm
    if pore < 5.0:
        return (
            f"Tight pores ({pore} nm) create kinetic trapping: target enters, "
            f"encounters dense binding field before diffusing back out. "
            f"Residence time inside cage >> open solution. "
            f"Effective on-rate enhanced by confinement."
        )
    elif pore < 15.0:
        return (
            f"Moderate pores ({pore} nm) partially confine target after entry. "
            f"Multiple binding encounters before exit. Mild kinetic enhancement."
        )
    return ""


def _describe_cooperativity(primary: RecognitionChemistry,
                              complement: RecognitionChemistry,
                              target, all_donors: set) -> str:
    """Describe what cooperativity the mixed pocket creates."""
    target_hsab = ""
    if target.electronic and target.electronic.hardness_softness:
        target_hsab = target.electronic.hardness_softness

    primary_donors = set(primary.donor_atoms)
    complement_donors = set(complement.donor_atoms)
    new_donors = complement_donors - primary_donors

    parts = []

    if target_hsab == "borderline" and "S" in all_donors and ("O" in all_donors or "N" in all_donors):
        parts.append(
            f"{target.identity} is borderline HSAB — mixed hard+soft donors "
            f"({'/'.join(sorted(all_donors))}) match its ambivalent character "
            f"better than either donor type alone"
        )

    if new_donors:
        parts.append(
            f"Complement adds {'/'.join(sorted(new_donors))} donors not present in primary"
        )

    if primary.type != complement.type:
        parts.append(
            f"Mixed modality ({primary.type} + {complement.type}) provides "
            f"redundant recognition — if one fails, other still captures"
        )

    return ". ".join(parts) if parts else "Mixed pocket with complementary donor atoms."
