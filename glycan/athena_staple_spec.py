"""
glycan/athena_staple_spec.py -- ATHENA-compatible staple modification list for pulldown.

Translates a glycan pulldown design into a concrete DNA origami staple
modification spec. Does NOT require ATHENA at runtime — produces a spec
that ATHENA or a human can consume to generate the final staple order.

Output: a list of StapleModification entries specifying:
  - Which staple positions get sugar-click-PEG extensions
  - Which get biotin-TEG for magnetic bead coupling
  - Which get PEG-passivation (block non-specific binding)
  - IDT-orderable modification strings
  - Estimated cost per staple type

Usage:
    from glycan.athena_staple_spec import design_origami_pulldown
    spec = design_origami_pulldown(
        sugar="Man", click_position="C1", click_chemistry="CuAAC",
        peg_n=160, bead_diameter_nm=50,
        cage_type="octahedron", edge_length_nm=30, edge_type="DX",
    )
    print(spec.summary)
    for s in spec.order_sheet:
        print(s)

Cage geometries from ATHENA/pyDAEDALUS conventions.
Cost data from IDT custom oligonucleotide pricing (2024).
"""

from dataclasses import dataclass, field
from typing import Optional
import math


# ── Cage geometry database ──────────────────────────────────────────────
# Exterior staple counts per cage type from ATHENA wireframe designs.
# These are the staple termini pointing OUTWARD from the cage surface.
# Source: Veneziano 2016 Science 352:1534 (ATHENA), Douglas 2009 Nature.

@dataclass
class CageGeometry:
    name: str
    display_name: str
    n_exterior_staples: int       # total staple termini on outer surface
    n_interior_staples: int       # total staple termini inside cage
    n_faces: int
    face_pore_nm: float           # approximate face pore diameter
    outer_diameter_nm: float      # approximate outer diameter
    surface_area_nm2: float       # approximate outer surface area
    notes: str

CAGE_GEOMETRIES: dict[str, CageGeometry] = {
    "tetrahedron_DX_30": CageGeometry(
        "tetrahedron_DX_30", "Tetrahedron (DX, 30 nm edge)",
        n_exterior_staples=60, n_interior_staples=60,
        n_faces=4, face_pore_nm=8, outer_diameter_nm=25,
        surface_area_nm2=1950,
        notes="Smallest cage. Good for small-target capture. Limited exterior sites.",
    ),
    "octahedron_DX_30": CageGeometry(
        "octahedron_DX_30", "Octahedron (DX, 30 nm edge)",
        n_exterior_staples=120, n_interior_staples=120,
        n_faces=8, face_pore_nm=10, outer_diameter_nm=35,
        surface_area_nm2=3100,
        notes="Standard diagnostic cage. Good balance of sites and pore size.",
    ),
    "octahedron_DX_50": CageGeometry(
        "octahedron_DX_50", "Octahedron (DX, 50 nm edge)",
        n_exterior_staples=200, n_interior_staples=200,
        n_faces=8, face_pore_nm=16, outer_diameter_nm=58,
        surface_area_nm2=8660,
        notes="Large cage. More exterior sites. Higher cost.",
    ),
    "octahedron_6HB_30": CageGeometry(
        "octahedron_6HB_30", "Octahedron (6HB, 30 nm edge)",
        n_exterior_staples=180, n_interior_staples=180,
        n_faces=8, face_pore_nm=6, outer_diameter_nm=35,
        surface_area_nm2=3100,
        notes="Stiffer edges. Smaller pores (better size exclusion). More staples per edge.",
    ),
    "icosahedron_DX_30": CageGeometry(
        "icosahedron_DX_30", "Icosahedron (DX, 30 nm edge)",
        n_exterior_staples=300, n_interior_staples=300,
        n_faces=20, face_pore_nm=6, outer_diameter_nm=48,
        surface_area_nm2=7790,
        notes="Most exterior sites. Highest sugar display capacity. Most expensive.",
    ),
}


# ── Modification types ──────────────────────────────────────────────────

@dataclass
class StapleModification:
    """A single staple modification entry for the IDT order sheet."""
    role: str                     # "sugar", "magnetic", "passivation", "reporter", "unmodified"
    modification_5prime: str      # IDT modification string at 5' end
    modification_3prime: str      # IDT modification string at 3' end
    extension_sequence: str       # additional bases for overhang (if needed)
    cost_per_staple_usd: float
    notes: str

    @property
    def idt_order_string(self) -> str:
        """Format for IDT custom oligo order (modification + extension)."""
        parts = []
        if self.modification_5prime:
            parts.append(f"5'-{self.modification_5prime}")
        if self.extension_sequence:
            parts.append(f"ext: {self.extension_sequence}")
        if self.modification_3prime:
            parts.append(f"3'-{self.modification_3prime}")
        return " | ".join(parts) if parts else "unmodified"


# Standard modification templates
def _sugar_click_modification(sugar: str, click_chemistry: str, peg_n: int) -> StapleModification:
    """Sugar-click-PEG extension on 3' overhang."""
    # Click handle on the staple overhang:
    # CuAAC: 5'-azide on sugar, alkyne on staple overhang
    # SPAAC: DBCO on staple, azide on sugar (no copper needed)
    if click_chemistry == "SPAAC":
        mod_3p = "DBCO-TEG"
        click_note = "SPAAC (copper-free); DBCO-TEG on staple 3', azide-PEG-sugar added post-assembly"
        cost = 150.0  # DBCO modification ~$100-150 from IDT
    else:  # CuAAC
        mod_3p = "alkyne"
        click_note = "CuAAC; alkyne on staple 3', azide-PEG-sugar added post-assembly with Cu catalyst"
        cost = 75.0  # alkyne modification ~$50-75

    # Overhang sequence: 20-nt poly-T spacer before click handle
    # Provides physical separation between cage surface and sugar
    extension = "T" * 20

    return StapleModification(
        role="sugar",
        modification_5prime="",
        modification_3prime=mod_3p,
        extension_sequence=extension,
        cost_per_staple_usd=cost,
        notes=f"{sugar}-PEG{peg_n}-{click_chemistry}; {click_note}",
    )


def _magnetic_modification() -> StapleModification:
    """Biotin-TEG for streptavidin-magnetic bead coupling."""
    return StapleModification(
        role="magnetic",
        modification_5prime="Biotin-TEG",
        modification_3prime="",
        extension_sequence="",
        cost_per_staple_usd=35.0,  # Biotin-TEG ~$25-50 from IDT
        notes="Biotin-TEG at 5'; binds streptavidin on magnetic bead; Kd ~10^-15 M",
    )


def _passivation_modification() -> StapleModification:
    """PEG-passivation to block non-specific binding."""
    return StapleModification(
        role="passivation",
        modification_5prime="",
        modification_3prime="",
        extension_sequence="T" * 10 + "AAAA",  # short overhang for PEG coupling
        cost_per_staple_usd=15.0,  # standard oligo, no special mod
        notes="Poly-T overhang + PEG-NHS post-assembly; blocks serum protein adsorption",
    )


def _reporter_modification(reporter: str = "FAM") -> StapleModification:
    """Fluorescent reporter for binding confirmation."""
    costs = {"FAM": 50.0, "Cy3": 75.0, "Cy5": 85.0}
    return StapleModification(
        role="reporter",
        modification_5prime=reporter,
        modification_3prime="",
        extension_sequence="",
        cost_per_staple_usd=costs.get(reporter, 75.0),
        notes=f"{reporter} fluorophore; signal quench on target binding or post-pulldown quantification",
    )


# ── Pulldown configuration presets ──────────────────────────────────────

@dataclass
class PulldownConfig:
    """Ratio of staple types for pulldown application."""
    sugar_fraction: float         # fraction of exterior staples for sugar display
    magnetic_fraction: float      # fraction for biotin-magnetic handles
    passivation_fraction: float   # fraction for PEG-passivation
    reporter_fraction: float      # fraction for fluorescent reporters

    def validate(self):
        total = self.sugar_fraction + self.magnetic_fraction + self.passivation_fraction + self.reporter_fraction
        assert abs(total - 1.0) < 0.01, f"Fractions must sum to 1.0, got {total}"


# Standard presets from NanoCage plan
PULLDOWN_PRESETS: dict[str, PulldownConfig] = {
    "cell_capture": PulldownConfig(
        sugar_fraction=0.50,
        magnetic_fraction=0.30,
        passivation_fraction=0.15,
        reporter_fraction=0.05,
    ),
    "high_avidity": PulldownConfig(
        sugar_fraction=0.70,
        magnetic_fraction=0.20,
        passivation_fraction=0.10,
        reporter_fraction=0.0,
    ),
    "diagnostic": PulldownConfig(
        sugar_fraction=0.40,
        magnetic_fraction=0.30,
        passivation_fraction=0.20,
        reporter_fraction=0.10,
    ),
    "environmental": PulldownConfig(
        sugar_fraction=0.60,
        magnetic_fraction=0.30,
        passivation_fraction=0.10,
        reporter_fraction=0.0,
    ),
}


# ── Main output dataclass ──────────────────────────────────────────────

@dataclass
class OrigamiPulldownSpec:
    """Complete ATHENA-compatible specification for DNA origami pulldown."""
    # Design identity
    cage_type: str
    cage_geometry: CageGeometry
    sugar: str
    click_position: str
    click_chemistry: str
    peg_n: int
    config_preset: str

    # Staple assignments
    n_sugar_staples: int
    n_magnetic_staples: int
    n_passivation_staples: int
    n_reporter_staples: int
    n_unmodified_staples: int

    # Modification specs
    sugar_mod: StapleModification
    magnetic_mod: StapleModification
    passivation_mod: StapleModification
    reporter_mod: Optional[StapleModification]

    # Cost estimate
    modified_staple_cost_usd: float
    unmodified_staple_cost_usd: float
    scaffold_cost_usd: float
    total_staple_cost_usd: float
    sugar_peg_reagent_cost_usd: float
    magnetic_bead_cost_usd: float
    total_estimated_cost_usd: float

    # Performance
    sugar_spacing_nm: float       # average distance between sugar staples on surface
    estimated_sugars_per_cage: int
    effective_valency: int        # sugars that can engage receptors simultaneously

    notes: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        lines = [
            f"DNA Origami Pulldown: {self.cage_geometry.display_name}",
            f"  Sugar: {self.sugar} @ {self.click_position} via {self.click_chemistry}, PEG_{self.peg_n}",
            f"  Config: {self.config_preset}",
            f"  Staples: {self.n_sugar_staples} sugar + {self.n_magnetic_staples} magnetic + {self.n_passivation_staples} passivation + {self.n_reporter_staples} reporter",
            f"  Sugar spacing: ~{self.sugar_spacing_nm:.1f} nm",
            f"  Effective valency: {self.effective_valency}",
            f"  Estimated cost: ${self.total_estimated_cost_usd:.0f}",
        ]
        return "\n".join(lines)

    @property
    def order_sheet(self) -> list[dict]:
        """Generate IDT-style order sheet entries."""
        sheet = []
        if self.n_sugar_staples > 0:
            sheet.append({
                "role": "sugar",
                "count": self.n_sugar_staples,
                "modification": self.sugar_mod.idt_order_string,
                "cost_each": self.sugar_mod.cost_per_staple_usd,
                "cost_total": self.n_sugar_staples * self.sugar_mod.cost_per_staple_usd,
                "notes": self.sugar_mod.notes,
            })
        if self.n_magnetic_staples > 0:
            sheet.append({
                "role": "magnetic",
                "count": self.n_magnetic_staples,
                "modification": self.magnetic_mod.idt_order_string,
                "cost_each": self.magnetic_mod.cost_per_staple_usd,
                "cost_total": self.n_magnetic_staples * self.magnetic_mod.cost_per_staple_usd,
                "notes": self.magnetic_mod.notes,
            })
        if self.n_passivation_staples > 0:
            sheet.append({
                "role": "passivation",
                "count": self.n_passivation_staples,
                "modification": self.passivation_mod.idt_order_string,
                "cost_each": self.passivation_mod.cost_per_staple_usd,
                "cost_total": self.n_passivation_staples * self.passivation_mod.cost_per_staple_usd,
                "notes": self.passivation_mod.notes,
            })
        if self.n_reporter_staples > 0 and self.reporter_mod is not None:
            sheet.append({
                "role": "reporter",
                "count": self.n_reporter_staples,
                "modification": self.reporter_mod.idt_order_string,
                "cost_each": self.reporter_mod.cost_per_staple_usd,
                "cost_total": self.n_reporter_staples * self.reporter_mod.cost_per_staple_usd,
                "notes": self.reporter_mod.notes,
            })
        sheet.append({
            "role": "unmodified",
            "count": self.n_unmodified_staples,
            "modification": "none (standard staples)",
            "cost_each": 8.0,
            "cost_total": self.n_unmodified_staples * 8.0,
            "notes": "Interior staples + unassigned exterior. Standard desalted oligos from IDT.",
        })
        sheet.append({
            "role": "scaffold",
            "count": 1,
            "modification": "M13mp18 scaffold strand (7249 nt)",
            "cost_each": 200.0,
            "cost_total": 200.0,
            "notes": "Bayou Biolabs or NEB M13mp18 ssDNA. One scaffold per cage design.",
        })
        return sheet


# ── Design function ─────────────────────────────────────────────────────

def design_origami_pulldown(
    sugar: str,
    click_position: str = "C1",
    click_chemistry: str = "SPAAC",
    peg_n: int = 160,
    bead_diameter_nm: float = 50.0,
    cage_type: str = "octahedron_DX_30",
    config_preset: str = "cell_capture",
    reporter: str = "FAM",
    receptor_density_per_um2: float = 1000.0,
) -> OrigamiPulldownSpec:
    """Design a complete DNA origami pulldown cage.

    Args:
        sugar: sugar to display (e.g., "Man", "Gal", "Neu5Ac")
        click_position: OH position for click handle (e.g., "C1", "C9")
        click_chemistry: "CuAAC" or "SPAAC" (copper-free)
        peg_n: PEG units between sugar and staple
        bead_diameter_nm: magnetic bead diameter
        cage_type: key into CAGE_GEOMETRIES
        config_preset: key into PULLDOWN_PRESETS
        reporter: fluorophore for reporter staples ("FAM", "Cy3", "Cy5")
        receptor_density_per_um2: target cell receptor density

    Returns:
        OrigamiPulldownSpec with complete staple modification assignments.
    """
    if cage_type not in CAGE_GEOMETRIES:
        raise ValueError(f"Unknown cage type: '{cage_type}'. Available: {sorted(CAGE_GEOMETRIES.keys())}")
    if config_preset not in PULLDOWN_PRESETS:
        raise ValueError(f"Unknown preset: '{config_preset}'. Available: {sorted(PULLDOWN_PRESETS.keys())}")

    cage = CAGE_GEOMETRIES[cage_type]
    config = PULLDOWN_PRESETS[config_preset]
    config.validate()

    n_ext = cage.n_exterior_staples
    n_int = cage.n_interior_staples

    # Assign exterior staples by fraction
    n_sugar = int(n_ext * config.sugar_fraction)
    n_magnetic = int(n_ext * config.magnetic_fraction)
    n_reporter = int(n_ext * config.reporter_fraction)
    n_passivation = n_ext - n_sugar - n_magnetic - n_reporter  # remainder

    # All interior staples are unmodified (no binders inside for pulldown)
    n_unmodified = n_int

    # Build modification specs
    sugar_mod = _sugar_click_modification(sugar, click_chemistry, peg_n)
    magnetic_mod = _magnetic_modification()
    passivation_mod = _passivation_modification()
    reporter_mod = _reporter_modification(reporter) if n_reporter > 0 else None

    # Cost calculation
    modified_cost = (
        n_sugar * sugar_mod.cost_per_staple_usd
        + n_magnetic * magnetic_mod.cost_per_staple_usd
        + n_passivation * passivation_mod.cost_per_staple_usd
        + (n_reporter * reporter_mod.cost_per_staple_usd if reporter_mod else 0)
    )
    unmodified_cost = n_unmodified * 8.0  # ~$8 per standard desalted oligo from IDT
    scaffold_cost = 200.0  # M13mp18 scaffold
    sugar_peg_reagent = n_sugar * 5.0  # ~$5 per sugar-PEG-azide/DBCO conjugate
    magnetic_bead = 50.0  # one vial streptavidin magnetic beads

    total_cost = modified_cost + unmodified_cost + scaffold_cost + sugar_peg_reagent + magnetic_bead

    # Sugar spacing: how far apart are sugar staples on the cage surface
    if n_sugar > 0:
        area_per_sugar = cage.surface_area_nm2 / n_sugar
        sugar_spacing = math.sqrt(area_per_sugar)
    else:
        sugar_spacing = float('inf')

    # Effective valency: how many sugars can engage cell receptors simultaneously
    # Limited by contact patch when cage sits on cell surface
    linker_reach_nm = peg_n * 0.35 / 10  # PEG contour / 10 (nm)
    R_cage = cage.outer_diameter_nm / 2
    if cage.outer_diameter_nm < 2 * linker_reach_nm:
        contact_area = 2 * math.pi * R_cage ** 2
    else:
        contact_area = 2 * math.pi * R_cage * linker_reach_nm

    sugars_in_contact = max(1, int(n_sugar * contact_area / cage.surface_area_nm2))
    n_receptors = max(1, int(contact_area / 1e6 * receptor_density_per_um2))
    effective_valency = min(sugars_in_contact, n_receptors)

    notes = [
        f"Cage: {cage.display_name}",
        f"Sugar display: {n_sugar} staples x {sugar}-PEG{peg_n}-{click_chemistry}",
        f"Post-assembly conjugation: add {sugar}-PEG-azide to alkyne/DBCO staples",
        f"Magnetic pulldown: {n_magnetic} biotin staples -> SA-magnetic beads ({bead_diameter_nm}nm)",
    ]
    if click_chemistry == "CuAAC":
        notes.append("CuAAC requires Cu(I) catalyst; remove Cu before cell contact (EDTA wash)")
    else:
        notes.append("SPAAC is copper-free; biocompatible for live cell capture")

    return OrigamiPulldownSpec(
        cage_type=cage_type,
        cage_geometry=cage,
        sugar=sugar,
        click_position=click_position,
        click_chemistry=click_chemistry,
        peg_n=peg_n,
        config_preset=config_preset,
        n_sugar_staples=n_sugar,
        n_magnetic_staples=n_magnetic,
        n_passivation_staples=n_passivation,
        n_reporter_staples=n_reporter,
        n_unmodified_staples=n_unmodified,
        sugar_mod=sugar_mod,
        magnetic_mod=magnetic_mod,
        passivation_mod=passivation_mod,
        reporter_mod=reporter_mod,
        modified_staple_cost_usd=round(modified_cost, 2),
        unmodified_staple_cost_usd=round(unmodified_cost, 2),
        scaffold_cost_usd=scaffold_cost,
        total_staple_cost_usd=round(modified_cost + unmodified_cost + scaffold_cost, 2),
        sugar_peg_reagent_cost_usd=round(sugar_peg_reagent, 2),
        magnetic_bead_cost_usd=magnetic_bead,
        total_estimated_cost_usd=round(total_cost, 2),
        sugar_spacing_nm=round(sugar_spacing, 1),
        estimated_sugars_per_cage=n_sugar,
        effective_valency=effective_valency,
        notes=notes,
    )


# ── Convenience ─────────────────────────────────────────────────────────

def list_cage_types() -> list[str]:
    return sorted(CAGE_GEOMETRIES.keys())

def list_presets() -> list[str]:
    return sorted(PULLDOWN_PRESETS.keys())
