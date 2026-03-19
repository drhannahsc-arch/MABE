"""
glycan/daedalus_bridge.py -- Bridge between pyDAEDALUS and MABE pulldown pipeline.

Runs pyDAEDALUS to generate real staple sequences for wireframe DNA origami
cages, then assigns sugar-click-PEG, biotin, passivation, and reporter
modifications to specific staples.

Two modes:
  1. LIVE: Runs pyDAEDALUS DX_cage_design on a PLY file → real staple sequences
  2. CSV:  Reads a pre-generated pyDAEDALUS staples CSV file

Output: DAEDALUSDesign with real sequences + modification assignments,
compatible with athena_staple_spec.py for cost estimation and IDT ordering.

Requires: pyDAEDALUS repo cloned at a known path (set via PYDAEDALUS_PATH
environment variable or passed as argument). If pyDAEDALUS is not available,
falls back to the geometry-only estimates in athena_staple_spec.py.

Usage:
    from glycan.daedalus_bridge import design_from_ply, design_from_csv
    design = design_from_csv("staples_octahedron.csv")
    print(design.summary)
    for s in design.modified_staples:
        print(f"{s.name}: {s.role} {s.sequence[:30]}...")

    # Or from PLY:
    design = design_from_ply("octahedron.ply", pydaedalus_path="/path/to/pyDAEDALUS")

Staple naming convention from pyDAEDALUS:
    {cage}_{min_nt}_{staple_id}-{scaffold_pos}-{type}
    type = V (vertex staple, longer, at vertices with polyT loops)
           E (edge staple, shorter, spans edge midpoints)
    M13_(...) = scaffold strand (not a staple)
"""

from dataclasses import dataclass, field
from typing import Optional
import os
import csv
import math


# ── Parsed staple ───────────────────────────────────────────────────────

@dataclass
class ParsedStaple:
    """A single staple parsed from pyDAEDALUS output."""
    name: str                     # full name from CSV (e.g., "tet_31_1-236-V")
    sequence: str                 # DNA sequence
    length: int                   # number of nucleotides
    staple_type: str              # "V" (vertex) or "E" (edge)
    staple_id: int                # numeric ID extracted from name
    scaffold_position: int        # scaffold position where staple binds

    @property
    def is_vertex(self) -> bool:
        return self.staple_type == "V"

    @property
    def is_edge(self) -> bool:
        return self.staple_type == "E"


@dataclass
class ModifiedStaple:
    """A staple with assigned modification for pulldown."""
    name: str
    sequence: str
    length: int
    staple_type: str
    role: str                     # "sugar", "magnetic", "passivation", "reporter", "unmodified"
    modification_5prime: str      # IDT modification string
    modification_3prime: str
    extension_3prime: str         # additional bases for overhang
    cost_usd: float
    note: str

    @property
    def idt_order_string(self) -> str:
        """Full IDT order: 5'mod + sequence + extension + 3'mod."""
        parts = []
        if self.modification_5prime:
            parts.append(f"/5{self.modification_5prime}/")
        parts.append(self.sequence)
        if self.extension_3prime:
            parts.append(self.extension_3prime)
        if self.modification_3prime:
            parts.append(f"/3{self.modification_3prime}/")
        return "".join(parts)

    @property
    def total_length(self) -> int:
        return self.length + len(self.extension_3prime)


# ── DAEDALUS design output ──────────────────────────────────────────────

@dataclass
class DAEDALUSDesign:
    """Complete DAEDALUS cage design with modification assignments."""
    cage_name: str
    n_staples: int
    n_vertex_staples: int
    n_edge_staples: int
    scaffold_length: int
    scaffold_sequence: str

    # Modification assignments
    modified_staples: list[ModifiedStaple]
    n_sugar: int
    n_magnetic: int
    n_passivation: int
    n_reporter: int
    n_unmodified: int

    # Cost
    total_cost_usd: float

    notes: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        lines = [
            f"DAEDALUS Design: {self.cage_name}",
            f"  Staples: {self.n_staples} ({self.n_vertex_staples}V + {self.n_edge_staples}E)",
            f"  Scaffold: {self.scaffold_length} nt",
            f"  Modifications: {self.n_sugar} sugar + {self.n_magnetic} magnetic + {self.n_passivation} passivation + {self.n_reporter} reporter + {self.n_unmodified} unmod",
            f"  Estimated cost: ${self.total_cost_usd:.0f}",
        ]
        return "\n".join(lines)

    @property
    def idt_order_sheet(self) -> list[dict]:
        """Generate IDT-format order sheet with real sequences."""
        sheet = []
        for ms in self.modified_staples:
            sheet.append({
                "name": ms.name,
                "role": ms.role,
                "sequence": ms.idt_order_string,
                "length": ms.total_length,
                "cost_usd": ms.cost_usd,
                "note": ms.note,
            })
        # Add scaffold
        sheet.append({
            "name": "M13mp18_scaffold",
            "role": "scaffold",
            "sequence": f"M13mp18 ({self.scaffold_length} nt)",
            "length": self.scaffold_length,
            "cost_usd": 200.0,
            "note": "Bayou Biolabs or NEB M13mp18 ssDNA",
        })
        return sheet


# ── CSV parser ──────────────────────────────────────────────────────────

def parse_staples_csv(csv_path: str) -> tuple[list[ParsedStaple], str, int]:
    """Parse a pyDAEDALUS staples CSV file.

    Returns:
        (staples, scaffold_sequence, scaffold_length)
    """
    staples = []
    scaffold_seq = ""

    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line == "None,None":
                continue

            parts = line.split(",", 1)
            if len(parts) != 2:
                continue

            name, sequence = parts[0].strip(), parts[1].strip()

            if name.startswith("M13"):
                scaffold_seq = sequence
                continue

            # Parse staple type from name: ...-V or ...-E
            if "-V," in line or name.endswith("-V") or "-V" in name.split(",")[0]:
                stype = "V"
            elif "-E," in line or name.endswith("-E") or "-E" in name.split(",")[0]:
                stype = "E"
            else:
                stype = "U"  # unknown

            # Extract staple_id and scaffold_position from name
            # Format: {cage}_{min_nt}_{id}-{pos}-{type}
            try:
                name_parts = name.rsplit("-", 2)
                if len(name_parts) >= 2:
                    staple_id = int(name_parts[0].rsplit("_", 1)[-1])
                    scaffold_pos = int(name_parts[1])
                else:
                    staple_id = len(staples)
                    scaffold_pos = 0
            except (ValueError, IndexError):
                staple_id = len(staples)
                scaffold_pos = 0

            staples.append(ParsedStaple(
                name=name,
                sequence=sequence,
                length=len(sequence),
                staple_type=stype,
                staple_id=staple_id,
                scaffold_position=scaffold_pos,
            ))

    return staples, scaffold_seq, len(scaffold_seq)


# ── Modification assignment ─────────────────────────────────────────────

# Modification costs (from IDT 2024 pricing)
_MOD_COSTS = {
    "sugar_SPAAC": 150.0,   # DBCO-TEG at 3'
    "sugar_CuAAC": 75.0,    # alkyne at 3'
    "magnetic": 35.0,        # Biotin-TEG at 5'
    "passivation": 15.0,     # standard oligo + PEG-NHS post-assembly
    "reporter_FAM": 50.0,
    "reporter_Cy3": 75.0,
    "reporter_Cy5": 85.0,
    "unmodified": 8.0,       # standard desalted oligo
}

# Sugar-click extension: 20-nt poly-T spacer before click handle
_SUGAR_EXTENSION = "T" * 20


def assign_modifications(
    staples: list[ParsedStaple],
    sugar: str = "Man",
    click_chemistry: str = "SPAAC",
    sugar_fraction: float = 0.50,
    magnetic_fraction: float = 0.30,
    passivation_fraction: float = 0.15,
    reporter_fraction: float = 0.05,
    reporter: str = "FAM",
    prefer_vertex_for_sugar: bool = True,
) -> list[ModifiedStaple]:
    """Assign modifications to parsed staples.

    Strategy:
    - Vertex staples (V) are preferred for sugar modifications because they're
      longer (50-78 nt), have accessible 3' ends, and sit at cage vertices
      where they project outward.
    - Edge staples (E) are shorter (20 nt) and better for passivation or
      left unmodified.
    - Magnetic (biotin) goes on a mix of V and E staples.

    Args:
        staples: parsed staple list from CSV
        sugar: sugar to display
        click_chemistry: "SPAAC" or "CuAAC"
        sugar_fraction: fraction of staples for sugar display
        magnetic_fraction: fraction for biotin-magnetic
        passivation_fraction: fraction for PEG-passivation
        reporter_fraction: fraction for fluorescent reporter
        prefer_vertex_for_sugar: if True, assign sugars to vertex staples first
        reporter: fluorophore type

    Returns:
        List of ModifiedStaple with assignments.
    """
    n_total = len(staples)
    n_sugar = int(n_total * sugar_fraction)
    n_magnetic = int(n_total * magnetic_fraction)
    n_reporter = int(n_total * reporter_fraction)
    n_passivation = n_total - n_sugar - n_magnetic - n_reporter

    # Sort: vertex staples first (better for sugar), then edge
    if prefer_vertex_for_sugar:
        sorted_staples = sorted(staples, key=lambda s: (0 if s.is_vertex else 1, s.staple_id))
    else:
        sorted_staples = list(staples)

    # Click chemistry setup
    if click_chemistry == "SPAAC":
        sugar_3prime = "DBCO-TEG"
        sugar_cost = _MOD_COSTS["sugar_SPAAC"]
        click_note = "SPAAC copper-free"
    else:
        sugar_3prime = "alkyne"
        sugar_cost = _MOD_COSTS["sugar_CuAAC"]
        click_note = "CuAAC (Cu catalyst required)"

    reporter_cost = _MOD_COSTS.get(f"reporter_{reporter}", 75.0)

    modified = []
    idx = 0

    # Assign sugar staples
    for _ in range(n_sugar):
        if idx >= len(sorted_staples):
            break
        s = sorted_staples[idx]
        modified.append(ModifiedStaple(
            name=s.name, sequence=s.sequence, length=s.length,
            staple_type=s.staple_type, role="sugar",
            modification_5prime="",
            modification_3prime=sugar_3prime,
            extension_3prime=_SUGAR_EXTENSION,
            cost_usd=sugar_cost,
            note=f"{sugar}-PEG-{click_chemistry}; {click_note}; attach {sugar}-PEG-azide post-assembly",
        ))
        idx += 1

    # Assign magnetic staples
    for _ in range(n_magnetic):
        if idx >= len(sorted_staples):
            break
        s = sorted_staples[idx]
        modified.append(ModifiedStaple(
            name=s.name, sequence=s.sequence, length=s.length,
            staple_type=s.staple_type, role="magnetic",
            modification_5prime="BiotinTEG",
            modification_3prime="",
            extension_3prime="",
            cost_usd=_MOD_COSTS["magnetic"],
            note="Biotin-TEG at 5'; binds streptavidin on magnetic bead",
        ))
        idx += 1

    # Assign reporter staples
    for _ in range(n_reporter):
        if idx >= len(sorted_staples):
            break
        s = sorted_staples[idx]
        modified.append(ModifiedStaple(
            name=s.name, sequence=s.sequence, length=s.length,
            staple_type=s.staple_type, role="reporter",
            modification_5prime=reporter,
            modification_3prime="",
            extension_3prime="",
            cost_usd=reporter_cost,
            note=f"{reporter} fluorophore for binding confirmation",
        ))
        idx += 1

    # Assign passivation staples
    for _ in range(n_passivation):
        if idx >= len(sorted_staples):
            break
        s = sorted_staples[idx]
        modified.append(ModifiedStaple(
            name=s.name, sequence=s.sequence, length=s.length,
            staple_type=s.staple_type, role="passivation",
            modification_5prime="",
            modification_3prime="",
            extension_3prime="T" * 10,
            cost_usd=_MOD_COSTS["passivation"],
            note="Poly-T overhang + PEG-NHS post-assembly; blocks non-specific binding",
        ))
        idx += 1

    return modified


# ── High-level design functions ─────────────────────────────────────────

def design_from_csv(
    csv_path: str,
    sugar: str = "Man",
    click_chemistry: str = "SPAAC",
    sugar_fraction: float = 0.50,
    magnetic_fraction: float = 0.30,
    passivation_fraction: float = 0.15,
    reporter_fraction: float = 0.05,
    reporter: str = "FAM",
) -> DAEDALUSDesign:
    """Design a pulldown cage from a pre-generated pyDAEDALUS CSV.

    Args:
        csv_path: path to staples CSV from pyDAEDALUS
        sugar: sugar to display
        click_chemistry: "SPAAC" or "CuAAC"
        sugar_fraction: fraction for sugar display (0-1)
        magnetic_fraction: fraction for magnetic handles (0-1)
        passivation_fraction: fraction for passivation (0-1)
        reporter_fraction: fraction for reporter (0-1)
        reporter: fluorophore type

    Returns:
        DAEDALUSDesign with real sequences and modification assignments.
    """
    staples, scaffold_seq, scaffold_len = parse_staples_csv(csv_path)

    if not staples:
        raise ValueError(f"No staples found in {csv_path}")

    modified = assign_modifications(
        staples, sugar, click_chemistry,
        sugar_fraction, magnetic_fraction, passivation_fraction, reporter_fraction,
        reporter,
    )

    cage_name = os.path.splitext(os.path.basename(csv_path))[0]
    n_v = sum(1 for s in staples if s.is_vertex)
    n_e = sum(1 for s in staples if s.is_edge)

    role_counts = {}
    for ms in modified:
        role_counts[ms.role] = role_counts.get(ms.role, 0) + 1

    total_cost = sum(ms.cost_usd for ms in modified) + 200.0  # + scaffold

    return DAEDALUSDesign(
        cage_name=cage_name,
        n_staples=len(staples),
        n_vertex_staples=n_v,
        n_edge_staples=n_e,
        scaffold_length=scaffold_len,
        scaffold_sequence=scaffold_seq,
        modified_staples=modified,
        n_sugar=role_counts.get("sugar", 0),
        n_magnetic=role_counts.get("magnetic", 0),
        n_passivation=role_counts.get("passivation", 0),
        n_reporter=role_counts.get("reporter", 0),
        n_unmodified=role_counts.get("unmodified", 0),
        total_cost_usd=round(total_cost, 2),
        notes=[
            f"Real staple sequences from pyDAEDALUS",
            f"Sugar: {sugar} via {click_chemistry}",
            f"Scaffold: M13mp18 ({scaffold_len} nt)",
        ],
    )


def design_from_ply(
    ply_path: str,
    pydaedalus_path: Optional[str] = None,
    min_len_nt: int = 31,
    sugar: str = "Man",
    click_chemistry: str = "SPAAC",
    sugar_fraction: float = 0.50,
    magnetic_fraction: float = 0.30,
    passivation_fraction: float = 0.15,
    reporter_fraction: float = 0.05,
    reporter: str = "FAM",
    output_dir: Optional[str] = None,
) -> DAEDALUSDesign:
    """Design a pulldown cage from a PLY geometry file.

    Runs pyDAEDALUS DX_cage_design, then assigns modifications.

    Args:
        ply_path: path to PLY geometry file
        pydaedalus_path: path to pyDAEDALUS repo (or set PYDAEDALUS_PATH env var)
        min_len_nt: minimum edge length in nucleotides
        sugar: sugar to display
        click_chemistry: "SPAAC" or "CuAAC"
        sugar_fraction, magnetic_fraction, passivation_fraction, reporter_fraction: fractions
        reporter: fluorophore type
        output_dir: where to write pyDAEDALUS output files

    Returns:
        DAEDALUSDesign with real sequences and modification assignments.
    """
    import sys

    # Find pyDAEDALUS
    if pydaedalus_path is None:
        pydaedalus_path = os.environ.get("PYDAEDALUS_PATH")
    if pydaedalus_path is None:
        raise ValueError(
            "pyDAEDALUS path not found. Set PYDAEDALUS_PATH environment variable "
            "or pass pydaedalus_path argument."
        )

    daedalus_dir = os.path.join(pydaedalus_path, "pyDAEDALUS")
    if not os.path.isdir(daedalus_dir):
        raise ValueError(f"pyDAEDALUS directory not found at {daedalus_dir}")

    # Add to path
    if daedalus_dir not in sys.path:
        sys.path.insert(0, daedalus_dir)
    ad_dir = os.path.join(daedalus_dir, "Automated_Design")
    if ad_dir not in sys.path:
        sys.path.insert(0, ad_dir)

    import matplotlib
    matplotlib.use("Agg")

    from Automated_Design.ply_to_input import ply_to_input
    from Automated_Design.DX_cage_design import DX_cage_design
    from Automated_Design.constants import M13_SCAF_SEQ

    # Parse PLY
    result = ply_to_input(ply_path, min_len_nt=min_len_nt)
    coords, edges, faces, edge_lengths, name, staple_name, singleXOs = result

    # Output directory
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), f"daedalus_{name}")
    os.makedirs(output_dir, exist_ok=True)

    # Run DX_cage_design
    DX_cage_design(
        coords, edges, faces, edge_lengths, name, staple_name,
        singleXOs, M13_SCAF_SEQ, "M13", False, output_dir, False,
        print_to_console=False,
    )

    # Find the staples CSV
    csv_files = [f for f in os.listdir(output_dir)
                 if f.startswith("staples_") and f.endswith(".csv")]
    if not csv_files:
        raise RuntimeError(f"pyDAEDALUS did not produce a staples CSV in {output_dir}")

    csv_path = os.path.join(output_dir, csv_files[0])

    return design_from_csv(
        csv_path, sugar, click_chemistry,
        sugar_fraction, magnetic_fraction, passivation_fraction, reporter_fraction,
        reporter,
    )


# ── Convenience ─────────────────────────────────────────────────────────

# PLY files bundled with pyDAEDALUS
STANDARD_CAGES = {
    "tetrahedron": "01_tetrahedron.ply",
    "cube": "02_cube.ply",
    "octahedron": "03_octahedron.ply",
    "dodecahedron": "04_dodecahedron.ply",
    "icosahedron": "05_icosahedron.ply",
    "cuboctahedron": "06_cuboctahedron.ply",
    "truncated_octahedron": "14_truncated_octahedron.ply",
    "truncated_icosahedron": "13_truncated_icosahedron.ply",
}

def list_standard_cages() -> list[str]:
    return sorted(STANDARD_CAGES.keys())
