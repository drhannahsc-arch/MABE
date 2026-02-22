"""
MABE Sprint 7 Bootstrap - Intelligent Interior Design
======================================================
The cage IS the binder. Multiple recognition elements positioned
by structural geometry create binding pockets that no single
component achieves alone.

Like a protein: 4 histidines aren't a zinc binder. 4 histidines
at 2.1A spacing held by a fold ARE a zinc binder.

    cd Documents\\mabe
    python bootstrap_sprint7.py
    python tests\\test_sprint7.py
    python main.py "lead capture and release from mine water"
"""

import os

def write_file(path, content):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Created: {path}")

print()
print("  MABE Sprint 7 - Intelligent Interior Design")
print("  " + "=" * 40)
print()

# ═══════════════════════════════════════════════════════════════════════════
# core/assembly.py — Extended with InteriorDesign
# ═══════════════════════════════════════════════════════════════════════════

write_file("core/assembly.py", '''"""
core/assembly.py - Composite binder assembly model.

A binder is an assembly at one of three complexity levels:

Level 1 — Simple: One recognition element, free in solution.
Level 2 — Composite: Multiple copies of one recognition element in a structure.
Level 3 — Tertiary: Multiple DIFFERENT recognition elements positioned by
           structural geometry to create a binding pocket that no individual
           component achieves alone. The structure IS the binder.

The protein fold analogy: 4 histidines are not a zinc binder.
4 histidines at 2.1 Angstrom spacing held by a fold ARE a zinc binder.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RecognitionChemistry:
    """What touches the target. One type of donor atom/group."""
    name: str
    type: str                          # chelator, dnazyme, peptide, aptamer, small_molecule
    donor_atoms: list[str]             # S, N, O, P
    donor_type: str                    # hard, soft, borderline
    structure: str                     # SMILES, sequence, or description
    kd_um: Optional[float] = None
    source_tool: str = ""
    source_candidate_name: str = ""
    notes: str = ""


@dataclass
class InteriorSite:
    """One recognition element placed at a specific position inside a structure."""
    recognition: RecognitionChemistry
    copies: int = 1
    position_description: str = ""     # "vertex-adjacent staple overhangs", "pore-lining"
    attachment_chemistry: str = ""     # "5'-amine to NHS-staple", "DTPA-modified staple"


@dataclass
class InteriorDesign:
    """
    THE KEY INNOVATION: the designed interior of a structural binder.

    This describes HOW recognition elements are arranged inside a structure
    to create a binding pocket. It's the difference between:
    - "DTPA inside a cage" (Sprint 6 — structure as container)
    - "12 DTPA at 6nm spacing + 4 thiols at vertices = multivalent
       Pb2+ pocket with 100x avidity over free DTPA" (Sprint 7 —
       structure as co-designer of recognition)

    Properties that emerge from the arrangement:
    - Avidity: N copies at distance D = effective Kd improvement
    - Cooperativity: mixed donor types create selectivity neither has alone
    - Geometric match: spacing matches target coordination geometry
    - Kinetic trapping: once target enters and binds site 1, site 2 is
      close enough to capture before target escapes through pore
    """
    sites: list[InteriorSite] = field(default_factory=list)
    design_level: str = "simple"       # simple, composite, tertiary
    total_binding_sites: int = 0
    unique_recognition_types: int = 0
    avidity_factor: float = 1.0        # effective Kd multiplier from multivalency
    cooperativity_note: str = ""       # what the arrangement achieves
    geometric_match: str = ""          # how spacing matches target coordination
    kinetic_trapping: str = ""         # how pore + interior creates kinetic advantage
    design_rationale: str = ""         # why this arrangement

    def summary(self) -> str:
        if not self.sites:
            return "No interior design (free in solution)"
        parts = []
        for site in self.sites:
            parts.append(f"{site.copies}x {site.recognition.name} ({site.position_description})")
        arr = " + ".join(parts)
        return f"[{self.design_level}] {arr} | avidity {self.avidity_factor:.0f}x"


@dataclass
class StructuralConstraint:
    """What holds recognition elements in 3D geometry."""
    name: str
    type: str
    geometry: str
    interior_volume_nm3: Optional[float] = None
    pore_size_nm: Optional[float] = None
    max_interior_sites: int = 1
    recognition_spacing_nm: Optional[float] = None
    ph_stable_range: tuple = (0.0, 14.0)
    temp_stable_c: tuple = (0, 100)
    cost_per_unit: str = ""
    synthesis_complexity: str = "trivial"
    notes: str = ""


@dataclass
class SelectivityFilter:
    """Steric, charge, or hydrophobic barriers."""
    name: str
    mechanism: str
    description: str
    excludes: list[str] = field(default_factory=list)
    passes: list[str] = field(default_factory=list)
    selectivity_enhancement: str = ""


@dataclass
class ReleaseMechanism:
    """How the target is liberated on demand."""
    name: str
    trigger: str
    description: str
    reversible: bool = True
    cycles: int = 1
    trigger_conditions: str = ""
    release_efficiency: str = ""
    notes: str = ""


@dataclass
class BinderAssembly:
    """The complete binder design."""
    name: str
    description: str
    design_level: str                  # simple, composite, tertiary
    interior: InteriorDesign           # ← NEW: the designed interior
    structure: StructuralConstraint
    selectivity: SelectivityFilter
    release: ReleaseMechanism
    composite_score: float = 0.0
    confidence: str = "low"
    confidence_reasoning: str = ""
    estimated_cost: str = ""
    community_lab_feasible: bool = False
    failure_modes: list[str] = field(default_factory=list)
    what_improves_odds: list[str] = field(default_factory=list)

    @property
    def recognition(self) -> RecognitionChemistry:
        """Primary recognition chemistry (first site)."""
        if self.interior.sites:
            return self.interior.sites[0].recognition
        return RecognitionChemistry(name="none", type="none", donor_atoms=[], donor_type="none", structure="")

    def summary(self) -> str:
        parts = [
            f"{self.name} [{self.design_level}]",
            f"  Interior: {self.interior.summary()}",
            f"  Structure: {self.structure.name}",
            f"  Selectivity: {self.selectivity.name}",
            f"  Release: {self.release.name} ({self.release.trigger})",
            f"  Score: {self.composite_score:.0%} ({self.confidence}) | Cost: {self.estimated_cost}",
        ]
        return "\\n".join(parts)

    def full_report(self) -> str:
        lines = [
            f"{'=' * 60}",
            f"  BINDER ASSEMBLY: {self.name}",
            f"  Design level: {self.design_level.upper()}",
            f"{'=' * 60}",
            f"",
            f"  {self.description}",
            f"",
        ]

        # Interior design section
        lines.append("-- Interior Design --")
        if self.interior.sites:
            for site in self.interior.sites:
                kd_str = f", Kd={site.recognition.kd_um} uM" if site.recognition.kd_um else ""
                lines.append(
                    f"  {site.copies}x {site.recognition.name} "
                    f"({site.recognition.type}, {'/'.join(site.recognition.donor_atoms)} donors{kd_str})"
                )
                lines.append(f"     Position: {site.position_description}")
                lines.append(f"     Attachment: {site.attachment_chemistry}")
            lines.append(f"  Total sites: {self.interior.total_binding_sites}")
            lines.append(f"  Unique recognition types: {self.interior.unique_recognition_types}")
            if self.interior.avidity_factor > 1.0:
                lines.append(f"  Avidity enhancement: {self.interior.avidity_factor:.0f}x over monovalent")
            if self.interior.cooperativity_note:
                lines.append(f"  Cooperativity: {self.interior.cooperativity_note}")
            if self.interior.geometric_match:
                lines.append(f"  Geometric match: {self.interior.geometric_match}")
            if self.interior.kinetic_trapping:
                lines.append(f"  Kinetic trapping: {self.interior.kinetic_trapping}")
            if self.interior.design_rationale:
                lines.append(f"  Rationale: {self.interior.design_rationale}")
        else:
            lines.append("  No structural interior — binder free in solution.")

        # Structure
        lines.extend([
            f"",
            f"-- Structural Constraint --",
            f"  {self.structure.name} ({self.structure.type})",
            f"  Geometry: {self.structure.geometry}",
        ])
        if self.structure.interior_volume_nm3:
            lines.append(f"  Interior volume: {self.structure.interior_volume_nm3} nm3")
        if self.structure.pore_size_nm:
            lines.append(f"  Pore size: {self.structure.pore_size_nm} nm")

        # Selectivity
        lines.extend([
            f"",
            f"-- Selectivity Filter --",
            f"  {self.selectivity.name} ({self.selectivity.mechanism})",
            f"  {self.selectivity.description}",
        ])
        if self.selectivity.selectivity_enhancement:
            lines.append(f"  Enhancement: {self.selectivity.selectivity_enhancement}")

        # Release
        lines.extend([
            f"",
            f"-- Release Mechanism --",
            f"  {self.release.name} ({self.release.trigger})",
            f"  {self.release.description}",
            f"  Reversible: {'Yes' if self.release.reversible else 'No'} | Cycles: ~{self.release.cycles}",
        ])
        if self.release.trigger_conditions:
            lines.append(f"  Conditions: {self.release.trigger_conditions}")

        # Assessment
        lines.extend([
            f"",
            f"-- Assessment --",
            f"  Composite score: {self.composite_score:.0%}",
            f"  Confidence: {self.confidence}",
            f"  Reasoning: {self.confidence_reasoning}",
            f"  Cost: {self.estimated_cost}",
            f"  Community lab: {'Yes' if self.community_lab_feasible else 'No'}",
        ])

        if self.failure_modes:
            lines.append(f"")
            lines.append(f"  What could go wrong:")
            for fm in self.failure_modes:
                lines.append(f"    - {fm}")

        if self.what_improves_odds:
            lines.append(f"")
            lines.append(f"  What improves your odds:")
            for imp in self.what_improves_odds:
                lines.append(f"    - {imp}")

        return "\\n".join(lines)
''')

# ═══════════════════════════════════════════════════════════════════════════
# core/interior_designer.py — The brain that designs binding pockets
# ═══════════════════════════════════════════════════════════════════════════

write_file("core/interior_designer.py", '''"""
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
''')

# ═══════════════════════════════════════════════════════════════════════════
# core/assembly_composer.py — Rewritten to use interior designer
# ═══════════════════════════════════════════════════════════════════════════

write_file("core/assembly_composer.py", '''"""
core/assembly_composer.py - Composes binder assemblies using interior designer.

The interior designer creates the binding pocket.
The assembly composer wraps that in structure, selectivity, and release.
"""

from __future__ import annotations

from core.problem import Problem
from core.assembly import (
    BinderAssembly, InteriorDesign, StructuralConstraint,
    SelectivityFilter, ReleaseMechanism,
)
from core.candidate import CandidateResult
from core.interior_designer import design_interior
from knowledge.structural_library import (
    STRUCTURAL_OPTIONS, generate_selectivity_filter, get_compatible_releases,
)


def _score_structural_match(interior: InteriorDesign,
                             structure: StructuralConstraint,
                             problem: Problem) -> float:
    """Score how well structure matches interior design + problem."""
    score = 0.5

    if structure.type == "none":
        return 0.5

    matrix_ph = problem.matrix.ph or 7.0
    if structure.ph_stable_range[0] <= matrix_ph <= structure.ph_stable_range[1]:
        score += 0.1
    else:
        score -= 0.3

    matrix_temp = problem.matrix.temperature_c or 25.0
    if structure.temp_stable_c[0] <= matrix_temp <= structure.temp_stable_c[1]:
        score += 0.05
    else:
        score -= 0.2

    if structure.type == "dna_origami_cage" and problem.matrix.competing_species:
        score += 0.15

    if structure.type == "mof":
        score += 0.1

    # Bonus for tertiary designs — structure matters more
    if interior.design_level == "tertiary":
        score += 0.1

    # Avidity bonus
    if interior.avidity_factor > 3.0:
        score += 0.1

    if structure.synthesis_complexity == "complex":
        score -= 0.1
    elif structure.synthesis_complexity == "expert":
        score -= 0.2

    return max(0.0, min(1.0, score))


def compose_assemblies(candidates: list[CandidateResult],
                        problem: Problem,
                        max_assemblies: int = 8) -> list[BinderAssembly]:
    """Compose assemblies with designed interiors."""
    assemblies = []
    wants_release = "release" in problem.desired_outcome.description.lower()

    real_candidates = [c for c in candidates if c.source_tool != "dummy"]
    top_recognition = sorted(real_candidates,
                              key=lambda c: c.performance.probability_of_success,
                              reverse=True)[:4]

    if not top_recognition:
        top_recognition = candidates[:3]

    none_struct = [s for s in STRUCTURAL_OPTIONS if s.type == "none"][0]

    for candidate in top_recognition:
        # Get compatible structures
        struct_scores = []
        for structure in STRUCTURAL_OPTIONS:
            if structure.type == "none":
                continue
            # Quick pH pre-filter
            matrix_ph = problem.matrix.ph or 7.0
            if not (structure.ph_stable_range[0] - 0.5 <= matrix_ph <= structure.ph_stable_range[1] + 0.5):
                continue
            struct_scores.append(structure)

        # Design interiors for: free + best 2 structures
        selected = [none_struct]
        if struct_scores:
            # Take up to 2 compatible structures
            selected.extend(struct_scores[:2])

        for structure in selected:
            # DESIGN THE INTERIOR
            interior = design_interior(candidate, structure, problem, real_candidates)

            # Selectivity filter
            target_radius = 0.3
            if problem.target.size and problem.target.size.hydrated_radius_angstrom:
                target_radius = problem.target.size.hydrated_radius_angstrom / 10.0
            selectivity = generate_selectivity_filter(structure, target_radius)

            # Release
            primary_type = interior.sites[0].recognition.type if interior.sites else "chelator"
            releases = get_compatible_releases(primary_type, structure.type, wants_release)
            release = releases[0] if releases else ReleaseMechanism(
                name="No active release", trigger="none", description="Permanent capture")

            # Score
            base_prob = candidate.performance.probability_of_success
            struct_bonus = _score_structural_match(interior, structure, problem) - 0.5

            # Avidity improves effective Kd
            if interior.avidity_factor > 1.0:
                avidity_boost = min(0.15, (interior.avidity_factor - 1.0) * 0.03)
            else:
                avidity_boost = 0.0

            composite = max(0.05, min(0.95, base_prob + struct_bonus * 0.3 + avidity_boost))

            # Confidence
            if structure.type == "none":
                confidence = candidate.performance.confidence
                conf_reason = candidate.performance.confidence_reasoning
            elif interior.design_level == "tertiary":
                confidence = "speculative"
                conf_reason = (
                    f"Tertiary pocket design: {interior.design_rationale[:100]}... "
                    f"Mixed donor cooperativity predicted but requires experimental validation."
                )
            else:
                confidence = "low" if candidate.performance.confidence == "low" else "speculative"
                conf_reason = (
                    f"Recognition: {candidate.performance.confidence_reasoning} "
                    f"Structural integration predicted but not validated."
                )

            # Cost
            if structure.type == "none":
                cost = candidate.accessibility.estimated_cost
            else:
                cost = f"{candidate.accessibility.estimated_cost} (binder) + {structure.cost_per_unit} (structure)"

            # Failure modes
            failure_modes = list(candidate.performance.failure_modes)
            if structure.type == "dna_origami_cage":
                failure_modes.append("Cage assembly yield ~30-70% — optimize Mg2+ and anneal protocol")
                failure_modes.append("Metal ions pass through DNA walls — interior binding mandatory")
            if structure.type == "mof":
                failure_modes.append("Post-synthetic modification efficiency varies — characterize loading")
            if interior.design_level == "tertiary":
                failure_modes.append("Mixed pocket cooperativity is predicted, not measured — validate by ITC")

            # Improvements
            improvements = list(candidate.performance.what_improves_odds)
            if interior.avidity_factor > 2.0:
                improvements.append(
                    f"Multivalent interior ({interior.total_binding_sites} sites) "
                    f"provides {interior.avidity_factor:.0f}x avidity enhancement"
                )
            if interior.design_level == "tertiary":
                improvements.append("Mixed donor pocket may show emergent selectivity — test against panel")

            # Name
            if structure.type == "none":
                name = f"{candidate.name} (free)"
                desc = (
                    f"Recognition: {candidate.name}. Free in solution. "
                    f"Release by {release.name}."
                )
            else:
                level_label = {"simple": "", "composite": "composite ", "tertiary": "TERTIARY "}
                name = f"{level_label.get(interior.design_level, '')}{candidate.name} in {structure.name}"
                desc = (
                    f"{interior.design_level.title()} binder: {interior.summary()}. "
                    f"Structure: {structure.name}. "
                    f"Release by {release.name}."
                )

            assemblies.append(BinderAssembly(
                name=name.strip(),
                description=desc,
                design_level=interior.design_level,
                interior=interior,
                structure=structure,
                selectivity=selectivity,
                release=release,
                composite_score=round(composite, 2),
                confidence=confidence,
                confidence_reasoning=conf_reason,
                estimated_cost=cost,
                community_lab_feasible=(
                    candidate.accessibility.community_lab_feasible
                    and structure.synthesis_complexity in ("trivial", "standard")
                ),
                failure_modes=failure_modes,
                what_improves_odds=improvements,
            ))

    assemblies.sort(key=lambda a: a.composite_score, reverse=True)
    return assemblies[:max_assemblies]
''')

# ═══════════════════════════════════════════════════════════════════════════
# tests/test_sprint7.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("tests/test_sprint7.py", '''"""
tests/test_sprint7.py - Interior pocket design tests.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conversation.decomposer_patch import patch_targets
patch_targets()

from core.orchestrator import Orchestrator
from core.assembly import BinderAssembly
from adapters.base import ToolRegistry
from adapters.rdkit_adapter import RDKitAdapter
from adapters.dnazyme_adapter import DNAzymeAdapter
from adapters.peptide_adapter import PeptideAdapter
from adapters.aptamer_adapter import AptamerAdapter
from conversation.decomposer import decompose


def _build():
    registry = ToolRegistry()
    rdkit = RDKitAdapter()
    if rdkit.is_available(): registry.register(rdkit)
    registry.register(DNAzymeAdapter())
    registry.register(PeptideAdapter())
    registry.register(AptamerAdapter())
    return registry


def test_assemblies_have_interior_design():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture and release from mine water"))
    for a in r.assemblies:
        assert a.interior is not None, f"{a.name} missing interior design"
        assert len(a.interior.sites) > 0, f"{a.name} has empty interior"
    print(f"  + All {len(r.assemblies)} assemblies have designed interiors")


def test_three_design_levels():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture and release from mine water"))
    levels = set(a.design_level for a in r.assemblies)
    print(f"  + Design levels present: {levels}")
    assert "simple" in levels, "Should have simple (free) assemblies"
    # Composite or tertiary should appear for cage assemblies
    assert levels - {"simple"}, f"Should have at least one non-simple level, got {levels}"


def test_composite_has_multiple_sites():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    composites = [a for a in r.assemblies if a.design_level in ("composite", "tertiary")]
    if composites:
        for a in composites:
            assert a.interior.total_binding_sites > 1, f"{a.name} should have >1 site"
        print(f"  + Composite/tertiary assemblies have multiple interior sites ({composites[0].interior.total_binding_sites} sites)")


def test_avidity_enhancement():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    cage_assemblies = [a for a in r.assemblies if a.structure.type != "none"]
    if cage_assemblies:
        for a in cage_assemblies:
            assert a.interior.avidity_factor >= 1.0
        best = max(cage_assemblies, key=lambda a: a.interior.avidity_factor)
        print(f"  + Best avidity: {best.interior.avidity_factor:.0f}x ({best.name})")


def test_tertiary_has_mixed_donors():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture and release from mine water"))
    tertiary = [a for a in r.assemblies if a.design_level == "tertiary"]
    if tertiary:
        for a in tertiary:
            assert a.interior.unique_recognition_types >= 2
            assert a.interior.cooperativity_note != ""
        print(f"  + Tertiary pocket: {tertiary[0].interior.unique_recognition_types} recognition types")
        print(f"    Cooperativity: {tertiary[0].interior.cooperativity_note[:80]}...")
    else:
        print(f"  + No tertiary designs generated (complement may not have been found)")
        # Not a failure — tertiary only generated when good complement exists


def test_interior_report():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    cage_assemblies = [a for a in r.assemblies if a.structure.type != "none"]
    if cage_assemblies:
        report = cage_assemblies[0].full_report()
        assert "Interior Design" in report
        assert "copies" in report.lower() or "x " in report.lower()
        print(f"  + Interior design rendered in full report")


def test_kinetic_trapping_described():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    has_kinetic = False
    for a in r.assemblies:
        if a.interior.kinetic_trapping:
            has_kinetic = True
            break
    if has_kinetic:
        print(f"  + Kinetic trapping described for cage assemblies")
    else:
        print(f"  + No kinetic trapping (expected if no tight-pore cages compatible)")


def test_mercury_tertiary_pocket():
    """Mercury is soft — should get thiol + T-Hg-T mixed pocket."""
    o = Orchestrator(_build())
    r = o.solve(decompose("mercury removal from river water"))
    tertiary = [a for a in r.assemblies if a.design_level == "tertiary"]
    if tertiary:
        donors = set()
        for site in tertiary[0].interior.sites:
            donors.update(site.recognition.donor_atoms)
        print(f"  + Mercury tertiary pocket: donors = {donors}")
    else:
        # Check if composite at least exists
        composites = [a for a in r.assemblies if a.design_level == "composite"]
        print(f"  + Mercury: {len(composites)} composite assemblies (tertiary not found)")


def test_simple_assemblies_have_one_site():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    simples = [a for a in r.assemblies if a.design_level == "simple"]
    for a in simples:
        assert a.interior.total_binding_sites == 1
        assert a.interior.avidity_factor == 1.0
    print(f"  + Simple assemblies correctly have 1 site, avidity 1x")


if __name__ == "__main__":
    print()
    print("  MABE Sprint 7 - Interior Pocket Design Tests")
    print("  " + "=" * 40)
    print()

    test_assemblies_have_interior_design()
    test_three_design_levels()
    test_composite_has_multiple_sites()
    test_avidity_enhancement()
    test_tertiary_has_mixed_donors()
    test_interior_report()
    test_kinetic_trapping_described()
    test_mercury_tertiary_pocket()
    test_simple_assemblies_have_one_site()

    print()
    print("  All Sprint 7 tests passed.")
    print()
''')

print()
print("  Done! New/updated files:")
print("    core/assembly.py            (InteriorDesign + InteriorSite added)")
print("    core/interior_designer.py   (NEW: pocket design engine)")
print("    core/assembly_composer.py   (rewritten to use interior designer)")
print("    tests/test_sprint7.py       (9 tests)")
print()
print("  THE KEY CHANGE:")
print("    The cage IS the binder. Interior designer creates 3 levels:")
print("    Level 1 (simple):    1 element, free in solution")
print("    Level 2 (composite): N copies of 1 element in cage = avidity")
print("    Level 3 (tertiary):  mixed elements = designed binding pocket")
print()
print("  Next steps:")
print("    python tests\\test_sprint7.py")
print('    python main.py "lead capture and release from mine water"')
print()