"""
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
        return "\n".join(parts)

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

        return "\n".join(lines)
