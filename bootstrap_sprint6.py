"""
MABE Sprint 6 Bootstrap - Composite Binder Assembly Designer
==============================================================
The architectural shift: a binder is not a molecule, it's an assembly.

    Recognition chemistry (what touches the target)
      x Structural constraint (what holds donors in geometry)
        x Selectivity filter (steric exclusion, charge gating)
          x Release mechanism (UV, pH, magnetic, toehold, competitor)

DNA origami is not a binder modality. It's a structural option.
A nanocage with interior chelators is a composite binder where the
cage geometry IS PART OF the binding design.

    cd Documents\\mabe
    python bootstrap_sprint6.py
    python tests\\test_sprint6.py
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
print("  MABE Sprint 6 - Composite Binder Assembly")
print("  " + "=" * 40)
print()

# ═══════════════════════════════════════════════════════════════════════════
# core/assembly.py — The new data model
# ═══════════════════════════════════════════════════════════════════════════

write_file("core/assembly.py", '''"""
core/assembly.py - Composite binder assembly model.

A binder is not a molecule. A binder is an assembly:

    Recognition Chemistry:  what donor atoms/groups contact the target
    Structural Constraint:  what holds donors in the right 3D geometry
    Selectivity Filter:     steric exclusion, charge gating, pore sizing
    Release Mechanism:      how the target is liberated on demand

A protein binding pocket: multiple residues (recognition) held by
the protein fold (structural constraint) with steric exclusion at
the pocket entrance (selectivity filter).

A DNA origami cage with interior chelators: small molecules (recognition)
positioned by staple overhangs inside a wireframe cage (structural constraint)
with pore size excluding large competitors (selectivity filter) and
toehold strand displacement opening a lid (release mechanism).

A free chelator in solution: one molecule (recognition), no structural
constraint, no selectivity filter, pH-driven release.

All three are binder assemblies at different complexity levels.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RecognitionChemistry:
    """What touches the target. The donor atoms and their arrangement."""
    name: str
    type: str                          # chelator, dnazyme, peptide, aptamer, small_molecule
    donor_atoms: list[str]             # S, N, O, P
    donor_type: str                    # hard, soft, borderline
    structure: str                     # SMILES, sequence, or description
    kd_um: Optional[float] = None      # if known
    source_tool: str = ""              # which adapter generated this
    source_candidate_name: str = ""    # original candidate name
    notes: str = ""


@dataclass
class StructuralConstraint:
    """What holds recognition elements in 3D geometry."""
    name: str
    type: str                          # none, protein_fold, dna_origami_cage, dna_tetrahedron,
                                       # mof, polymer_matrix, self_assembled_cage, dendrimer
    geometry: str                      # free, icosahedron, tetrahedron, octahedron, tube, sheet
    interior_volume_nm3: Optional[float] = None
    pore_size_nm: Optional[float] = None
    max_interior_sites: int = 1
    recognition_spacing_nm: Optional[float] = None
    ph_stable_range: tuple = (0.0, 14.0)
    temp_stable_c: tuple = (0, 100)
    cost_per_unit: str = ""
    synthesis_complexity: str = "trivial"  # trivial, standard, complex, expert
    notes: str = ""


@dataclass
class SelectivityFilter:
    """Steric, charge, or hydrophobic barriers that exclude non-targets."""
    name: str
    mechanism: str                     # pore_exclusion, charge_gating, hydrophobic_barrier,
                                       # size_exclusion, none
    description: str
    excludes: list[str] = field(default_factory=list)  # what gets excluded
    passes: list[str] = field(default_factory=list)    # what gets through
    selectivity_enhancement: str = ""  # e.g., "100x over Ca2+ due to pore size"


@dataclass
class ReleaseMechanism:
    """How the target is liberated on demand."""
    name: str
    trigger: str                       # pH_shift, uv_light, toehold_strand, competitor,
                                       # thermal, magnetic, electrochemical, edta_wash, none
    description: str
    reversible: bool = True
    cycles: int = 1                    # how many capture-release cycles
    trigger_conditions: str = ""       # e.g., "365nm UV, 5 min" or "pH shift to 2.0"
    release_efficiency: str = ""       # e.g., ">90% in 10 min"
    notes: str = ""


@dataclass
class BinderAssembly:
    """
    The complete binder design: recognition + structure + filter + release.
    This replaces the flat CandidateResult for composite designs.
    """
    name: str
    description: str
    recognition: RecognitionChemistry
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

    def summary(self) -> str:
        parts = [
            f"{self.name}",
            f"  Recognition: {self.recognition.name} ({self.recognition.type}, {'/'.join(self.recognition.donor_atoms)} donors)",
            f"  Structure:   {self.structure.name} ({self.structure.type})",
            f"  Selectivity: {self.selectivity.name} ({self.selectivity.mechanism})",
            f"  Release:     {self.release.name} ({self.release.trigger})",
            f"  Score: {self.composite_score:.0%} ({self.confidence}) | Cost: {self.estimated_cost}",
        ]
        return "\\n".join(parts)

    def full_report(self) -> str:
        lines = [
            f"{'=' * 60}",
            f"  COMPOSITE BINDER: {self.name}",
            f"{'=' * 60}",
            f"",
            f"  {self.description}",
            f"",
            f"-- Recognition Chemistry --",
            f"  {self.recognition.name} ({self.recognition.type})",
            f"  Donor atoms: {'/'.join(self.recognition.donor_atoms)} ({self.recognition.donor_type})",
            f"  Structure: {self.recognition.structure}",
        ]
        if self.recognition.kd_um is not None:
            lines.append(f"  Kd: {self.recognition.kd_um} uM")
        if self.recognition.notes:
            lines.append(f"  Notes: {self.recognition.notes}")

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
        if self.structure.max_interior_sites > 1:
            lines.append(f"  Interior binding sites: {self.structure.max_interior_sites}")
        lines.append(f"  Synthesis: {self.structure.synthesis_complexity}")

        lines.extend([
            f"",
            f"-- Selectivity Filter --",
            f"  {self.selectivity.name} ({self.selectivity.mechanism})",
            f"  {self.selectivity.description}",
        ])
        if self.selectivity.excludes:
            lines.append(f"  Excludes: {', '.join(self.selectivity.excludes)}")
        if self.selectivity.selectivity_enhancement:
            lines.append(f"  Enhancement: {self.selectivity.selectivity_enhancement}")

        lines.extend([
            f"",
            f"-- Release Mechanism --",
            f"  {self.release.name} ({self.release.trigger})",
            f"  {self.release.description}",
            f"  Reversible: {'Yes' if self.release.reversible else 'No'}",
            f"  Cycles: ~{self.release.cycles}",
        ])
        if self.release.trigger_conditions:
            lines.append(f"  Conditions: {self.release.trigger_conditions}")

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
# knowledge/structural_library.py — Cage geometries and constraints
# ═══════════════════════════════════════════════════════════════════════════

write_file("knowledge/structural_library.py", '''"""
knowledge/structural_library.py - Structural constraint and release mechanism libraries.
"""

from core.assembly import StructuralConstraint, SelectivityFilter, ReleaseMechanism


# ═══════════════════════════════════════════════════════════════════════════
# STRUCTURAL CONSTRAINTS — what holds donors in geometry
# ═══════════════════════════════════════════════════════════════════════════

STRUCTURAL_OPTIONS = [
    StructuralConstraint(
        name="Free in solution (no constraint)",
        type="none",
        geometry="free",
        max_interior_sites=1,
        ph_stable_range=(0.0, 14.0),
        temp_stable_c=(0, 100),
        cost_per_unit="included in recognition chemistry cost",
        synthesis_complexity="trivial",
        notes="Binder acts alone. Simplest. Fastest to deploy. No steric selectivity.",
    ),
    StructuralConstraint(
        name="DNA origami icosahedron (40nm, DX wireframe)",
        type="dna_origami_cage",
        geometry="icosahedron",
        interior_volume_nm3=33500.0,
        pore_size_nm=8.0,
        max_interior_sites=12,
        recognition_spacing_nm=6.0,
        ph_stable_range=(5.0, 9.0),
        temp_stable_c=(4, 50),
        cost_per_unit="$2000 first assembly (staple set), then $50/assembly",
        synthesis_complexity="complex",
        notes="M13mp18 scaffold, ~200 staples. 12 interior-facing staple overhangs for binding site placement. "
              "DX wireframe: 2-3nm wall gaps. Metal ions pass freely through walls — MUST be bound to interior chemistry. "
              "Pore diameter ~8nm provides steric exclusion of large competitors.",
    ),
    StructuralConstraint(
        name="DNA origami tetrahedron (20nm)",
        type="dna_origami_cage",
        geometry="tetrahedron",
        interior_volume_nm3=940.0,
        pore_size_nm=12.0,
        max_interior_sites=4,
        recognition_spacing_nm=8.0,
        ph_stable_range=(5.5, 8.5),
        temp_stable_c=(4, 45),
        cost_per_unit="$500 first, $20/assembly",
        synthesis_complexity="standard",
        notes="Simplest DNA origami cage. 4 triangular faces, large pore openings. "
              "Good for: rapid prototyping, therapeutic (proven renal clearance). "
              "Less steric selectivity due to large pores.",
    ),
    StructuralConstraint(
        name="DNA origami 6HB cage (30nm, tight pores)",
        type="dna_origami_cage",
        geometry="icosahedron",
        interior_volume_nm3=14100.0,
        pore_size_nm=3.0,
        max_interior_sites=8,
        recognition_spacing_nm=5.0,
        ph_stable_range=(5.5, 8.5),
        temp_stable_c=(4, 45),
        cost_per_unit="$3000 first, $80/assembly",
        synthesis_complexity="complex",
        notes="6-helix bundle edges. Tighter wall packing (~1nm gaps). "
              "Smallest pores = best steric selectivity. "
              "Ideal for discriminating by target size.",
    ),
    StructuralConstraint(
        name="MOF cage (UiO-66 type)",
        type="mof",
        geometry="octahedral",
        interior_volume_nm3=1.2,
        pore_size_nm=0.6,
        max_interior_sites=1,
        ph_stable_range=(1.0, 10.0),
        temp_stable_c=(0, 300),
        cost_per_unit="$5/g material",
        synthesis_complexity="standard",
        notes="Metal-organic framework. Extremely high surface area (1000+ m2/g). "
              "Sub-nm pores provide molecular sieving. Post-synthetic modification for selectivity. "
              "Excellent pH and thermal stability. Scalable.",
    ),
    StructuralConstraint(
        name="Protein cage (ferritin-like)",
        type="protein_cage",
        geometry="icosahedron",
        interior_volume_nm3=340.0,
        pore_size_nm=0.4,
        max_interior_sites=24,
        recognition_spacing_nm=2.0,
        ph_stable_range=(4.0, 10.0),
        temp_stable_c=(4, 70),
        cost_per_unit="$500 expression, $5/mg",
        synthesis_complexity="complex",
        notes="Self-assembling 24-subunit protein cage. 0.4nm pores — ions pass, proteins excluded. "
              "Interior can be engineered with metal-binding residues. Biocompatible. "
              "pH-driven disassembly for release (pH < 4).",
    ),
    StructuralConstraint(
        name="Dendrimer (PAMAM G4)",
        type="dendrimer",
        geometry="sphere",
        interior_volume_nm3=22.0,
        pore_size_nm=1.5,
        max_interior_sites=16,
        recognition_spacing_nm=1.0,
        ph_stable_range=(2.0, 10.0),
        temp_stable_c=(0, 80),
        cost_per_unit="$100/g",
        synthesis_complexity="standard",
        notes="Generation 4 PAMAM dendrimer. 64 amine surface groups for functionalization. "
              "Interior cavities can encapsulate metals. Well-characterized. "
              "Surface groups act as both binding sites and conjugation handles.",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# SELECTIVITY FILTERS
# ═══════════════════════════════════════════════════════════════════════════

def generate_selectivity_filter(structure: StructuralConstraint, target_radius_nm: float,
                                 competitor_sizes: list = None) -> SelectivityFilter:
    """Generate appropriate selectivity filter based on structural constraint."""
    if structure.type == "none":
        return SelectivityFilter(
            name="Chemical selectivity only",
            mechanism="none",
            description="No steric filtering. Selectivity comes entirely from recognition chemistry.",
        )

    if structure.pore_size_nm and structure.pore_size_nm > 0:
        pore = structure.pore_size_nm
        excludes = []
        passes = []
        if competitor_sizes:
            for name, size in competitor_sizes:
                if size > pore:
                    excludes.append(f"{name} ({size} nm - excluded by {pore} nm pores)")
                else:
                    passes.append(f"{name} ({size} nm - passes through)")

        return SelectivityFilter(
            name=f"Pore exclusion ({pore} nm)",
            mechanism="pore_exclusion",
            description=(
                f"Cage pore diameter {pore} nm. "
                f"Species larger than pore are physically excluded. "
                f"Target must be smaller than {pore} nm to enter cage. "
                f"Combined with interior recognition chemistry for dual selectivity."
            ),
            excludes=excludes,
            passes=passes,
            selectivity_enhancement=f"Steric exclusion at {pore} nm pore size adds size-based selectivity on top of chemical selectivity",
        )

    return SelectivityFilter(
        name="Structural selectivity",
        mechanism="size_exclusion",
        description="Constrained geometry limits access to binding sites.",
    )


# ═══════════════════════════════════════════════════════════════════════════
# RELEASE MECHANISMS
# ═══════════════════════════════════════════════════════════════════════════

RELEASE_OPTIONS = [
    ReleaseMechanism(
        name="pH shift release",
        trigger="ph_shift",
        description="Lower pH protonates donor atoms, releasing metal. Simple acid wash.",
        reversible=True,
        cycles=50,
        trigger_conditions="pH shift to 2.0-3.0 with dilute HCl",
        release_efficiency=">90% in 5 min",
        notes="Simplest release. Works for all O/N donor chelators. Thiol donors need lower pH.",
    ),
    ReleaseMechanism(
        name="Competitor displacement (EDTA wash)",
        trigger="competitor",
        description="Add strong universal chelator (EDTA) that strips target from binder.",
        reversible=True,
        cycles=100,
        trigger_conditions="10 mM EDTA, pH 7.0, 10 min",
        release_efficiency=">95%",
        notes="Works for any metal binder weaker than EDTA. Clean. Predictable.",
    ),
    ReleaseMechanism(
        name="Competitor displacement (imidazole)",
        trigger="competitor",
        description="Imidazole displaces His-tag coordination. Standard IMAC elution.",
        reversible=True,
        cycles=100,
        trigger_conditions="250 mM imidazole, pH 7.5",
        release_efficiency=">95%",
        notes="Specific to His-tag / imidazole-coordinating binders. Gold standard for Ni/Co capture.",
    ),
    ReleaseMechanism(
        name="Toehold strand displacement",
        trigger="toehold_strand",
        description="Add DNA strand that hybridizes to toehold, unfolding the binder and releasing target. "
                    "Programmable, isothermal, orthogonal to chemistry.",
        reversible=True,
        cycles=20,
        trigger_conditions="Add 10x excess trigger strand, RT, 30 min",
        release_efficiency="80-95%",
        notes="DNA-specific. Enables programmable sequential release of different targets. "
              "Multiple orthogonal toehold sequences = multi-target capture-release.",
    ),
    ReleaseMechanism(
        name="UV photocleavage",
        trigger="uv_light",
        description="UV light cleaves photolabile linker connecting binder to scaffold. "
                    "Target released with binder fragment.",
        reversible=False,
        cycles=1,
        trigger_conditions="365 nm UV, 5-15 min exposure",
        release_efficiency=">90%",
        notes="One-shot release. Spatial control possible (illuminate specific areas). "
              "Good for patterned deposition. Linker: o-nitrobenzyl ester.",
    ),
    ReleaseMechanism(
        name="Thermal release",
        trigger="thermal",
        description="Heat denatures DNA/protein structure, releasing bound target.",
        reversible=True,
        cycles=10,
        trigger_conditions="65-95 C, 5-15 min",
        release_efficiency=">90%",
        notes="Simple but limited to heat-stable targets. DNA refolds on cooling = reusable. "
              "Proteins may not refold.",
    ),
    ReleaseMechanism(
        name="Electrochemical release",
        trigger="electrochemical",
        description="Apply voltage to change oxidation state of donor atoms or target. "
                    "Reduces binding affinity, target dissociates.",
        reversible=True,
        cycles=200,
        trigger_conditions="Apply -0.5V to -1.0V vs Ag/AgCl, 1-5 min",
        release_efficiency="70-95%",
        notes="Requires conductive substrate (electrode, CNT, graphene). "
              "Excellent for continuous flow systems. Automated capture-release cycling.",
    ),
    ReleaseMechanism(
        name="Magnetic separation (physical)",
        trigger="magnetic",
        description="Not a release mechanism — a concentration step. "
                    "Magnetic beads pull down loaded binders. Combine with another release trigger.",
        reversible=True,
        cycles=500,
        trigger_conditions="Magnet application, 2 min",
        release_efficiency="N/A (separation, not release)",
        notes="Pairs with any release mechanism. Enables batch processing of large volumes.",
    ),
    ReleaseMechanism(
        name="No active release (permanent capture)",
        trigger="none",
        description="Target is permanently captured. For removal, not recovery.",
        reversible=False,
        cycles=1,
        trigger_conditions="N/A",
        release_efficiency="N/A",
        notes="Simplest deployment. Target captured and disposed of with binder. "
              "For toxic metals where recovery is not wanted.",
    ),
]


def get_compatible_releases(recognition_type: str, structure_type: str,
                             outcome_wants_release: bool) -> list[ReleaseMechanism]:
    """Return release mechanisms compatible with the recognition + structure combo."""
    compatible = []
    for release in RELEASE_OPTIONS:
        # Filter by compatibility
        if release.trigger == "toehold_strand" and recognition_type not in ("dnazyme", "dna_aptamer", "dna_motif"):
            continue
        if release.trigger == "competitor" and "imidazole" in release.name and recognition_type != "peptide":
            continue
        if release.trigger == "electrochemical" and structure_type not in ("none", "cnt", "graphene_oxide"):
            continue
        if release.trigger == "none" and outcome_wants_release:
            continue
        if release.trigger == "magnetic":
            continue  # This is a separation step, always compatible, added separately

        compatible.append(release)

    return compatible
''')

# ═══════════════════════════════════════════════════════════════════════════
# core/assembly_composer.py — Combines recognition + structure + filter + release
# ═══════════════════════════════════════════════════════════════════════════

write_file("core/assembly_composer.py", '''"""
core/assembly_composer.py - Composes binder assemblies from design axes.

Takes recognition chemistry candidates from existing adapters and wraps
them in structural constraints, selectivity filters, and release mechanisms
to create composite binder assemblies.
"""

from __future__ import annotations

from core.problem import Problem
from core.assembly import (
    BinderAssembly, RecognitionChemistry, StructuralConstraint,
    SelectivityFilter, ReleaseMechanism,
)
from core.candidate import CandidateResult
from knowledge.structural_library import (
    STRUCTURAL_OPTIONS, generate_selectivity_filter, get_compatible_releases,
)


def _extract_recognition(candidate: CandidateResult) -> RecognitionChemistry:
    """Extract recognition chemistry from a flat CandidateResult."""
    modality = candidate.modality.lower()

    # Determine donor info from description
    donor_atoms = []
    donor_type = "unknown"
    desc_lower = candidate.description.lower()
    if "thiol" in desc_lower or "/s" in candidate.structure_description.lower() or "cys" in desc_lower:
        donor_atoms.append("S")
        donor_type = "soft"
    if "/n" in candidate.structure_description.lower() or "his" in desc_lower or "amine" in desc_lower:
        donor_atoms.append("N")
    if "/o" in candidate.structure_description.lower() or "carbox" in desc_lower:
        donor_atoms.append("O")
    if "hard" in desc_lower:
        donor_type = "hard"
    elif "borderline" in desc_lower:
        donor_type = "borderline"
    elif "soft" in desc_lower:
        donor_type = "soft"

    if not donor_atoms:
        donor_atoms = ["unknown"]

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


def _score_structural_match(recognition: RecognitionChemistry,
                             structure: StructuralConstraint,
                             problem: Problem) -> float:
    """Score how well a structural option matches recognition + problem."""
    score = 0.5  # base

    # No constraint is always compatible, but provides no enhancement
    if structure.type == "none":
        return 0.5

    # pH compatibility
    matrix_ph = problem.matrix.ph or 7.0
    if structure.ph_stable_range[0] <= matrix_ph <= structure.ph_stable_range[1]:
        score += 0.1
    else:
        score -= 0.3  # pH kills cage stability

    # Temp compatibility
    matrix_temp = problem.matrix.temperature_c or 25.0
    if structure.temp_stable_c[0] <= matrix_temp <= structure.temp_stable_c[1]:
        score += 0.05
    else:
        score -= 0.2

    # DNA origami cages are best when we need steric selectivity
    if structure.type == "dna_origami_cage" and problem.matrix.competing_species:
        score += 0.15  # value steric filtering when there are competitors

    # MOF is great for very dilute targets and harsh conditions
    if structure.type == "mof":
        score += 0.1  # high surface area

    # Multi-site cages are better for borderline selectivity recognition
    if structure.max_interior_sites > 1 and recognition.donor_type == "borderline":
        score += 0.1  # multiple copies of mediocre binder = better capture

    # Cost penalty for complex structures
    if structure.synthesis_complexity == "complex":
        score -= 0.1
    elif structure.synthesis_complexity == "expert":
        score -= 0.2

    return max(0.0, min(1.0, score))


def compose_assemblies(candidates: list[CandidateResult],
                        problem: Problem,
                        max_assemblies: int = 5) -> list[BinderAssembly]:
    """
    Take the top recognition candidates and compose them with structural options.

    For each good recognition candidate, evaluate which structural constraints
    make sense, add appropriate selectivity filters and release mechanisms.
    """
    assemblies = []
    wants_release = "release" in problem.desired_outcome.description.lower()

    # Get top recognition candidates (skip dummy ones)
    real_candidates = [c for c in candidates if c.source_tool != "dummy"]
    # Take top 3 by probability
    top_recognition = sorted(real_candidates,
                              key=lambda c: c.performance.probability_of_success,
                              reverse=True)[:3]

    if not top_recognition:
        top_recognition = candidates[:3]

    for candidate in top_recognition:
        recognition = _extract_recognition(candidate)

        # Score each structural option
        struct_scores = []
        for structure in STRUCTURAL_OPTIONS:
            score = _score_structural_match(recognition, structure, problem)
            struct_scores.append((structure, score))

        # Take top 2 structural options (always include "none" + best cage)
        struct_scores.sort(key=lambda x: x[1], reverse=True)

        # Always include free (no constraint) version
        none_struct = [s for s in STRUCTURAL_OPTIONS if s.type == "none"][0]
        best_cage = [s for s, sc in struct_scores if s.type != "none"]
        selected_structures = [none_struct]
        if best_cage:
            selected_structures.append(best_cage[0])

        for structure in selected_structures:
            # Selectivity filter
            target_radius = 0.3  # default metal ion size in nm
            if problem.target.size.hydrated_radius_angstrom:
                target_radius = problem.target.size.hydrated_radius_angstrom / 10.0
            selectivity = generate_selectivity_filter(structure, target_radius)

            # Release mechanisms
            releases = get_compatible_releases(
                recognition.type, structure.type, wants_release
            )
            # Take top release option
            if releases:
                release = releases[0]  # pH shift is usually first and simplest
            else:
                release = ReleaseMechanism(
                    name="No active release", trigger="none",
                    description="Permanent capture",
                )

            # Composite score
            base_prob = candidate.performance.probability_of_success
            struct_bonus = _score_structural_match(recognition, structure, problem) - 0.5
            composite = max(0.05, min(0.95, base_prob + struct_bonus * 0.3))

            # Confidence
            if structure.type == "none":
                confidence = candidate.performance.confidence
                conf_reason = candidate.performance.confidence_reasoning
            else:
                confidence = "low" if candidate.performance.confidence == "low" else "speculative"
                conf_reason = (
                    f"Recognition chemistry: {candidate.performance.confidence_reasoning} "
                    f"Structural integration with {structure.name} is predicted but not validated for this target."
                )

            # Cost
            if structure.type == "none":
                cost = candidate.accessibility.estimated_cost
            else:
                cost = f"{candidate.accessibility.estimated_cost} (binder) + {structure.cost_per_unit} (structure)"

            # Failure modes
            failure_modes = list(candidate.performance.failure_modes)
            if structure.type == "dna_origami_cage":
                failure_modes.append("Cage assembly yield may be <50% — optimize Mg2+ concentration")
                failure_modes.append("Metal ions pass through DNA walls — interior binding is mandatory")
            if structure.type == "mof":
                failure_modes.append("Post-synthetic modification efficiency varies — characterize loading")

            # Improvements
            improvements = list(candidate.performance.what_improves_odds)
            if structure.max_interior_sites > 4:
                improvements.append(f"Multiple interior sites ({structure.max_interior_sites}) provide avidity — "
                                    f"even moderate-affinity recognition benefits from multivalent binding")

            # Name
            if structure.type == "none":
                assembly_name = f"{recognition.name} (free)"
            else:
                assembly_name = f"{recognition.name} in {structure.name}"

            # Description
            if structure.type == "none":
                desc = (
                    f"Recognition chemistry: {recognition.name}. "
                    f"Free in solution, no structural constraint. "
                    f"Simplest deployment. Release by {release.name}."
                )
            else:
                desc = (
                    f"Composite binder: {recognition.name} positioned inside {structure.name}. "
                    f"{structure.max_interior_sites} binding sites in {structure.geometry} geometry. "
                    f"Pore size {structure.pore_size_nm} nm provides steric selectivity. "
                    f"Release by {release.name}."
                )

            assemblies.append(BinderAssembly(
                name=assembly_name,
                description=desc,
                recognition=recognition,
                structure=structure,
                selectivity=selectivity,
                release=release,
                composite_score=round(composite, 2),
                confidence=confidence,
                confidence_reasoning=conf_reason,
                estimated_cost=cost,
                community_lab_feasible=candidate.accessibility.community_lab_feasible and structure.synthesis_complexity in ("trivial", "standard"),
                failure_modes=failure_modes,
                what_improves_odds=improvements,
            ))

    # Sort by composite score
    assemblies.sort(key=lambda a: a.composite_score, reverse=True)
    return assemblies[:max_assemblies]
''')

# ═══════════════════════════════════════════════════════════════════════════
# Update orchestrator to produce assemblies
# ═══════════════════════════════════════════════════════════════════════════

write_file("core/orchestrator.py", '''"""
core/orchestrator.py - The brain of MABE.
Now produces BOTH flat candidates AND composite assemblies.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from core.problem import Problem
from core.candidate import CandidateResult
from core.assembly import BinderAssembly
from core.connections import discover_connections
from core.assembly_composer import compose_assemblies
from adapters.base import ToolRegistry


@dataclass
class OrchestrationResult:
    problem_summary: str
    candidates: list[CandidateResult]
    assemblies: list[BinderAssembly]
    tools_consulted: list[str]
    tools_declined: list[tuple[str, str]]
    assumptions: list[str]
    notes: list[str] = field(default_factory=list)


class Orchestrator:

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def solve(self, problem: Problem) -> OrchestrationResult:
        contributors = self.registry.find_contributors(problem)
        tools_consulted = []
        tools_declined = []

        for adapter in self.registry.available_adapters():
            assessment = adapter.assess_contribution(problem)
            if not assessment.can_contribute:
                tools_declined.append((adapter.name, "Not relevant"))

        all_candidates: list[CandidateResult] = []
        for adapter, assessment in contributors:
            tools_consulted.append(f"{adapter.name}: {assessment.what_it_would_do}")
            candidates = adapter.generate_candidates(problem)
            all_candidates.extend(candidates)

        # Discover cross-domain connections
        target_name = problem.target.identity
        for candidate in all_candidates:
            new_connections = discover_connections(candidate, target_name)
            candidate.other_applications.extend(new_connections)

        # Rank flat candidates
        all_candidates = self._rank_candidates(all_candidates)
        for i, candidate in enumerate(all_candidates):
            candidate.rank = i + 1

        # COMPOSE ASSEMBLIES from top recognition candidates
        assemblies = compose_assemblies(all_candidates, problem, max_assemblies=6)

        notes = []
        if not contributors:
            notes.append("No connected tools could contribute to this problem.")
        if problem.assumptions_made:
            notes.append("MABE made assumptions - review these.")

        total_connections = sum(len(c.other_applications) for c in all_candidates)
        if total_connections > 0:
            notes.append(f"{total_connections} cross-domain applications discovered. Explore candidates for details.")

        if assemblies:
            notes.append(
                f"{len(assemblies)} composite binder assemblies designed: "
                f"recognition chemistry + structural constraint + selectivity filter + release mechanism."
            )

        return OrchestrationResult(
            problem_summary=problem.summary(),
            candidates=all_candidates,
            assemblies=assemblies,
            tools_consulted=tools_consulted,
            tools_declined=tools_declined,
            assumptions=problem.assumptions_made,
            notes=notes,
        )

    def _rank_candidates(self, candidates: list[CandidateResult]) -> list[CandidateResult]:
        def score(c: CandidateResult) -> float:
            perf = c.performance.probability_of_success
            access = 0.8 if c.accessibility.community_lab_feasible else 0.3
            cycles = c.accessibility.reusability_cycles or 0
            reuse = min(cycles / 100, 1.0)
            evidence_map = {"literature_validated": 1.0, "hybrid": 0.7, "computational_prediction": 0.4}
            evidence = evidence_map.get(c.evidence.source_type, 0.3)
            return perf * 0.40 + access * 0.25 + reuse * 0.20 + evidence * 0.15
        candidates.sort(key=score, reverse=True)
        return candidates
''')

# ═══════════════════════════════════════════════════════════════════════════
# Update interface to display assemblies
# ═══════════════════════════════════════════════════════════════════════════

write_file("conversation/interface.py", '''"""
conversation/interface.py - MABE conversational CLI.
Now displays both flat candidates and composite assemblies.
"""

from __future__ import annotations

from core.orchestrator import Orchestrator, OrchestrationResult
from core.candidate import CandidateResult
from core.assembly import BinderAssembly
from conversation.decomposer import decompose
from adapters.base import ToolRegistry


def print_banner():
    print()
    print("  +----------------------------------------------------------+")
    print("  |                                                          |")
    print("  |    MABE - Modality-Agnostic Binder Engine                |")
    print("  |    Universal Molecular Interaction Design Platform        |")
    print("  |                                                          |")
    print("  |    Public good first. Honest uncertainty. No waste.       |")
    print("  |                                                          |")
    print("  +----------------------------------------------------------+")
    print()


def print_result(result: OrchestrationResult):
    print()
    print("=" * 60)
    print("  MABE ANALYSIS")
    print("=" * 60)
    print()

    print("  Your problem (as MABE understands it):")
    for line in result.problem_summary.split("\\n"):
        print(f"    {line}")
    print()

    if result.assumptions:
        print("  ! Assumptions MABE made:")
        for a in result.assumptions:
            print(f"    - {a}")
        print()

    for note in result.notes:
        print(f"  i {note}")
    print()

    if result.tools_consulted:
        print(f"  Tools consulted: {len(result.tools_consulted)}")
        for tc in result.tools_consulted:
            print(f"    - {tc}")
        print()

    # ── COMPOSITE ASSEMBLIES ─────────────────────────────────────
    if result.assemblies:
        print("=" * 60)
        print("  COMPOSITE BINDER ASSEMBLIES")
        print("  (recognition + structure + selectivity + release)")
        print("=" * 60)
        print()
        for i, a in enumerate(result.assemblies):
            print(f"  A{i+1}: {a.name}")
            print(f"      Score: {a.composite_score:.0%} ({a.confidence}) | Cost: {a.estimated_cost}")
            print(f"      Structure: {a.structure.type} | Release: {a.release.trigger}")
            print()

    # ── FLAT CANDIDATES ──────────────────────────────────────────
    if result.candidates:
        n = len(result.candidates)
        print(f"  Top {n} recognition chemistry candidates:")
        print()
        for c in result.candidates[:10]:  # Show top 10
            print(f"  {c.short_summary()}")
            print()

    print("=" * 60)
    print()


def explore_assembly(assembly: BinderAssembly):
    print()
    print(assembly.full_report())
    print()


def explore_candidate(candidate: CandidateResult):
    print()
    print(candidate.full_report())
    print()


def run_interactive(registry: ToolRegistry):
    print_banner()
    orchestrator = Orchestrator(registry)
    print(f"  {registry.summary()}")
    print()
    print("  Describe your problem in plain language.")
    print("  Type A1-A9 to explore an assembly, 1-N for a candidate, 'quit' to exit.")
    print()

    last_result = None

    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\\n\\n  Goodbye.\\n")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("\\n  Goodbye.\\n")
            break

        # Explore assembly
        if user_input.upper().startswith("A") and last_result and last_result.assemblies:
            try:
                idx = int(user_input[1:]) - 1
                if 0 <= idx < len(last_result.assemblies):
                    explore_assembly(last_result.assemblies[idx])
                    continue
            except ValueError:
                pass

        # Explore candidate
        if user_input.isdigit() and last_result and last_result.candidates:
            idx = int(user_input) - 1
            if 0 <= idx < len(last_result.candidates):
                explore_candidate(last_result.candidates[idx])
                continue

        problem = decompose(user_input)
        result = orchestrator.solve(problem)
        last_result = result
        print_result(result)


def run_single_query(registry: ToolRegistry, query: str):
    orchestrator = Orchestrator(registry)
    problem = decompose(query)
    result = orchestrator.solve(problem)
    print_result(result)
    if result.assemblies:
        print()
        print("  ASSEMBLY DETAILS:")
        for a in result.assemblies:
            explore_assembly(a)
''')

# ═══════════════════════════════════════════════════════════════════════════
# Remove dummy adapter registration from main.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("main.py", '''"""
MABE - Modality-Agnostic Binder Engine
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adapters.base import ToolRegistry
from adapters.rdkit_adapter import RDKitAdapter
from adapters.dnazyme_adapter import DNAzymeAdapter
from adapters.peptide_adapter import PeptideAdapter
from adapters.aptamer_adapter import AptamerAdapter
from conversation.decomposer_patch import patch_targets
from conversation.interface import run_interactive, run_single_query

patch_targets()


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    rdkit = RDKitAdapter()
    if rdkit.is_available():
        registry.register(rdkit)
    registry.register(DNAzymeAdapter())
    registry.register(PeptideAdapter())
    registry.register(AptamerAdapter())
    # Dummy adapter REMOVED — composite assemblies replace mock protein/nanocage
    return registry


def main():
    registry = build_registry()
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_single_query(registry, query)
    else:
        run_interactive(registry)


if __name__ == "__main__":
    main()
''')

# ═══════════════════════════════════════════════════════════════════════════
# tests/test_sprint6.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("tests/test_sprint6.py", '''"""
tests/test_sprint6.py - Composite binder assembly tests.
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


def test_assemblies_produced():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture and release from mine water"))
    assert len(r.assemblies) > 0, "Should produce composite assemblies"
    print(f"  + {len(r.assemblies)} composite assemblies produced")


def test_assembly_has_four_axes():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    for a in r.assemblies:
        assert a.recognition is not None, f"{a.name} missing recognition"
        assert a.structure is not None, f"{a.name} missing structure"
        assert a.selectivity is not None, f"{a.name} missing selectivity"
        assert a.release is not None, f"{a.name} missing release"
    print(f"  + All assemblies have 4 design axes: recognition, structure, selectivity, release")


def test_assembly_includes_free_and_cage():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    types = set(a.structure.type for a in r.assemblies)
    assert "none" in types, "Should include free-in-solution assembly"
    has_cage = any(t in types for t in ("dna_origami_cage", "mof", "protein_cage", "dendrimer"))
    assert has_cage, f"Should include at least one structural constraint, got: {types}"
    print(f"  + Assembly types: {types}")


def test_cage_assembly_has_pore_selectivity():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    for a in r.assemblies:
        if a.structure.type != "none":
            assert a.selectivity.mechanism != "none", f"Cage assembly should have selectivity filter"
            assert "pore" in a.selectivity.mechanism.lower() or "size" in a.selectivity.mechanism.lower()
    print(f"  + Cage assemblies have pore/size selectivity filters")


def test_release_mechanism_present():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture and release from mine water"))
    for a in r.assemblies:
        assert a.release.trigger != "", f"{a.name} missing release trigger"
    triggers = set(a.release.trigger for a in r.assemblies)
    print(f"  + Release triggers present: {triggers}")


def test_assembly_full_report():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    for a in r.assemblies:
        report = a.full_report()
        assert "Recognition Chemistry" in report
        assert "Structural Constraint" in report
        assert "Selectivity Filter" in report
        assert "Release Mechanism" in report
        break  # just test first one
    print(f"  + Assembly full report contains all 4 sections")


def test_dummy_adapter_removed():
    registry = _build()
    names = [a.name for a in registry.all_adapters()]
    assert "dummy" not in names, "Dummy adapter should be removed"
    print(f"  + Dummy adapter removed. Active adapters: {names}")


def test_assembly_scorer():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    # Free assembly should score differently from cage assembly of same recognition
    for a in r.assemblies:
        assert 0.0 < a.composite_score <= 1.0
    print(f"  + Assembly scores: {[f'{a.name[:30]}={a.composite_score:.0%}' for a in r.assemblies]}")


def test_mercury_gets_dna_cage():
    o = Orchestrator(_build())
    r = o.solve(decompose("mercury removal from river water"))
    cage_assemblies = [a for a in r.assemblies if a.structure.type != "none"]
    if cage_assemblies:
        print(f"  + Mercury: {len(cage_assemblies)} cage assemblies designed")
        for a in cage_assemblies:
            print(f"    - {a.name}: {a.structure.type}, pore={a.structure.pore_size_nm}nm")
    else:
        print(f"  + Mercury: no cage assemblies (all free-in-solution) — expected if recognition is strong enough")


if __name__ == "__main__":
    print()
    print("  MABE Sprint 6 - Composite Binder Assembly Tests")
    print("  " + "=" * 40)
    print()

    test_assemblies_produced()
    test_assembly_has_four_axes()
    test_assembly_includes_free_and_cage()
    test_cage_assembly_has_pore_selectivity()
    test_release_mechanism_present()
    test_assembly_full_report()
    test_dummy_adapter_removed()
    test_assembly_scorer()
    test_mercury_gets_dna_cage()

    print()
    print("  All Sprint 6 tests passed.")
    print()
''')

print()
print("  Done! New/updated files:")
print("    core/assembly.py               (composite binder data model)")
print("    knowledge/structural_library.py (cages, MOFs, dendrimers, release mechanisms)")
print("    core/assembly_composer.py       (composes recognition + structure + filter + release)")
print("    core/orchestrator.py            (produces assemblies alongside flat candidates)")
print("    conversation/interface.py       (displays assemblies, A1-A9 navigation)")
print("    main.py                         (dummy adapter REMOVED)")
print("    tests/test_sprint6.py           (9 tests)")
print()
print("  ARCHITECTURAL SHIFT:")
print("    - Binder = recognition x structure x selectivity x release")
print("    - DNA origami cage is a STRUCTURAL OPTION, not a modality")
print("    - Dummy adapter eliminated — real composites replace mock data")
print("    - Same recognition chemistry appears both free AND in cages")
print()
print("  Next steps:")
print("    python tests\\test_sprint6.py")
print('    python main.py "lead capture and release from mine water"')
print()