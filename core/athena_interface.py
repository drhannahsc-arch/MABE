"""
core/athena_interface.py — ATHENA / pyDAEDALUS Interface

Translation layer between MABE's cage design and ATHENA/pyDAEDALUS
staple sequence generators.

ATHENA (Autonomous Tool for Hierarchical Exploration of NAnotechnology):
    - Input: PLY mesh of target polyhedron + edge type (DX/6HB)
    - Output: scaffold routing, staple sequences, crossover positions

pyDAEDALUS:
    - Input: target polyhedron specification
    - Output: DX wireframe staple sequences + scaffold routing

This module generates well-formed input specs and parses output.
Runs in SPEC MODE when ATHENA/pyDAEDALUS are not installed:
generates the spec files + documents what the tools would produce.

Does NOT:
    - Require ATHENA or pyDAEDALUS to be installed
    - Generate staple sequences itself (that's ATHENA's job)
    - Modify cage geometry (receives from dna_origami_pocket.py)
"""

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# ATHENA INPUT SPECIFICATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ATHENASpec:
    """Complete specification for ATHENA/pyDAEDALUS staple generation.

    This is the handoff document: everything ATHENA needs to generate
    staple sequences for a cage with interior module modifications.
    """
    # Cage geometry
    cage_id: str
    polyhedron: str
    edge_type: str               # "DX" or "6HB"
    edge_length_bp: int          # edge length in base pairs
    scaffold: str                # scaffold strand identity

    # PLY mesh (for ATHENA's mesh-based input)
    ply_vertices: list = field(default_factory=list)   # [(x,y,z), ...]
    ply_faces: list = field(default_factory=list)       # [(v0,v1,v2), ...]

    # Module modifications: staples that need 5' or 3' extensions
    modified_staples: list = field(default_factory=list)
    # list[ModifiedStaple]

    # Output expectations
    expected_staple_count: int = 0
    expected_scaffold_length_nt: int = 7249  # M13mp18

    # Tool configuration
    tool: str = "ATHENA"         # "ATHENA" or "pyDAEDALUS"
    mode: str = "spec"           # "spec" (generate input) or "run" (execute tool)

    def to_json(self):
        """Serialize to JSON for file output or tool handoff."""
        return json.dumps(asdict(self), indent=2, default=str)

    def to_ply(self):
        """Generate PLY mesh string for ATHENA input."""
        lines = [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(self.ply_vertices)}",
            "property float x",
            "property float y",
            "property float z",
            f"element face {len(self.ply_faces)}",
            "property list uchar int vertex_indices",
            "end_header",
        ]
        for v in self.ply_vertices:
            lines.append(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
        for f in self.ply_faces:
            lines.append(f"{len(f)} " + " ".join(str(i) for i in f))
        return "\n".join(lines)


@dataclass
class ModifiedStaple:
    """A staple that needs a functional module extension."""
    staple_id: str
    helix_id: int
    terminus: str                # "5prime" or "3prime"
    extension_type: str          # "linker+module"
    module_name: str
    module_smiles: str
    linker_type: str             # "PEG4", "C6-alkyl", etc.
    linker_length_nm: float
    conjugation: str             # "azide-SPAAC", "NHS-amine", etc.
    target_position_nm: tuple = (0, 0, 0)  # where the module should point
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# ATHENA OUTPUT PARSING (for when ATHENA is available)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ATHENAResult:
    """Parsed output from ATHENA/pyDAEDALUS."""
    scaffold_routing: str = ""           # scaffold sequence with routing
    staple_sequences: list = field(default_factory=list)  # [(staple_id, sequence), ...]
    crossover_positions: list = field(default_factory=list)
    interior_termini: list = field(default_factory=list)  # staple termini pointing inward
    exterior_termini: list = field(default_factory=list)  # staple termini pointing outward
    tool_version: str = ""
    success: bool = False
    error: str = ""


def parse_athena_output(output_path: str) -> ATHENAResult:
    """Parse ATHENA CSV/JSON output into ATHENAResult.

    ATHENA outputs:
        - scaffold_routing.csv: scaffold sequence with node assignments
        - staple_sequences.csv: staple ID, sequence, start/end positions
        - topology.json: crossover map + terminus classification

    This is a STUB — will be implemented when ATHENA is integrated.
    """
    result = ATHENAResult()
    result.error = f"ATHENA output parsing not yet implemented. Path: {output_path}"
    return result


def parse_daedalus_output(output_path: str) -> ATHENAResult:
    """Parse pyDAEDALUS output into ATHENAResult.

    pyDAEDALUS outputs:
        - .cndo file: connectivity and nucleotide data
        - Staple sequences in CSV format

    This is a STUB — will be implemented when pyDAEDALUS is integrated.
    """
    result = ATHENAResult()
    result.error = f"pyDAEDALUS output parsing not yet implemented. Path: {output_path}"
    return result


# ═══════════════════════════════════════════════════════════════════════════
# SPEC GENERATION: CagePreset + ModulePlacements → ATHENASpec
# ═══════════════════════════════════════════════════════════════════════════

def generate_athena_spec(cage, placements) -> ATHENASpec:
    """Generate ATHENA input specification from cage + module placements.

    Args:
        cage: CagePreset from dna_origami_pocket
        placements: list[ModulePlacement] from map_modules_to_staples

    Returns:
        ATHENASpec ready for file output or tool execution.
    """
    # Edge length in bp: edge_nm / 0.34 nm per bp, rounded to nearest multiple of 10.5
    # (DNA helical repeat = 10.5 bp)
    raw_bp = cage.edge_length_nm / 0.34
    edge_bp = int(round(raw_bp / 10.5) * 10.5)

    # Generate PLY mesh
    vertices, faces = _polyhedron_to_ply(cage.polyhedron, cage.edge_length_nm)

    # Build modified staple list
    modified = []
    for p in placements:
        modified.append(ModifiedStaple(
            staple_id=p.staple_id,
            helix_id=0,  # will be assigned by ATHENA
            terminus="3prime",  # convention: modules extend from 3' end
            extension_type="linker+module",
            module_name=p.module.name,
            module_smiles=p.module.smiles,
            linker_type=p.module.linker_type,
            linker_length_nm=p.linker_length_nm,
            conjugation=p.conjugation,
            target_position_nm=p.required_position_nm,
        ))

    spec = ATHENASpec(
        cage_id=cage.cage_id,
        polyhedron=cage.polyhedron,
        edge_type=cage.edge_type,
        edge_length_bp=edge_bp,
        scaffold=cage.scaffold,
        ply_vertices=vertices,
        ply_faces=faces,
        modified_staples=modified,
        expected_staple_count=cage.staple_count,
        expected_scaffold_length_nt=7249 if cage.scaffold == "M13mp18_7249" else 8064,
        tool="ATHENA" if cage.edge_type == "6HB" else "pyDAEDALUS",
        mode="spec",
    )

    return spec


def _polyhedron_to_ply(polyhedron, edge_nm):
    """Generate PLY vertex/face lists for a polyhedron.

    Returns (vertices, faces) where vertices are (x,y,z) in nm
    and faces are tuples of vertex indices.
    """
    from core.dna_origami_pocket import _polyhedron_topology

    raw_verts, edges, _ = _polyhedron_topology(polyhedron)

    # Scale to edge_nm
    # First compute current edge length
    if len(edges) > 0:
        v0, v1 = edges[0]
        current_edge = math.sqrt(sum(
            (raw_verts[v0][d] - raw_verts[v1][d])**2 for d in range(3)
        ))
        scale = edge_nm / current_edge if current_edge > 0.001 else edge_nm
    else:
        scale = edge_nm

    vertices = [(v[0]*scale, v[1]*scale, v[2]*scale) for v in raw_verts]

    # Generate triangular faces from edges
    # For simple polyhedra, find face cycles
    faces = _find_faces(vertices, edges)

    return vertices, faces


def _find_faces(vertices, edges):
    """Find triangular faces from edge list.

    For convex polyhedra, triangulate using adjacency.
    Simple approach: for each pair of edges sharing a vertex,
    check if closing edge exists → triangle.
    """
    edge_set = set()
    adj = {}
    for v1, v2 in edges:
        edge_set.add((min(v1,v2), max(v1,v2)))
        adj.setdefault(v1, set()).add(v2)
        adj.setdefault(v2, set()).add(v1)

    faces = set()
    for v1 in adj:
        for v2 in adj[v1]:
            for v3 in adj[v1]:
                if v2 < v3 and v3 in adj[v2]:
                    face = tuple(sorted([v1, v2, v3]))
                    faces.add(face)

    return [list(f) for f in faces]


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE: write spec files to disk
# ═══════════════════════════════════════════════════════════════════════════

def write_athena_files(spec: ATHENASpec, output_dir: str = "."):
    """Write ATHENA input files to disk.

    Generates:
        {cage_id}.ply         — mesh for ATHENA
        {cage_id}_spec.json   — full specification including modifications
        {cage_id}_README.txt  — human-readable summary

    Args:
        spec: ATHENASpec from generate_athena_spec()
        output_dir: directory to write files to

    Returns:
        dict of {filename: path} for generated files.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    files = {}

    # PLY mesh
    ply_path = os.path.join(output_dir, f"{spec.cage_id}.ply")
    with open(ply_path, 'w') as f:
        f.write(spec.to_ply())
    files["ply"] = ply_path

    # JSON spec
    json_path = os.path.join(output_dir, f"{spec.cage_id}_spec.json")
    with open(json_path, 'w') as f:
        f.write(spec.to_json())
    files["json"] = json_path

    # README
    readme_path = os.path.join(output_dir, f"{spec.cage_id}_README.txt")
    with open(readme_path, 'w') as f:
        f.write(f"MABE DNA Origami Pocket Design\n")
        f.write(f"{'=' * 40}\n\n")
        f.write(f"Cage: {spec.cage_id}\n")
        f.write(f"Polyhedron: {spec.polyhedron}\n")
        f.write(f"Edge type: {spec.edge_type}\n")
        f.write(f"Edge length: {spec.edge_length_bp} bp\n")
        f.write(f"Scaffold: {spec.scaffold}\n")
        f.write(f"Expected staples: {spec.expected_staple_count}\n")
        f.write(f"Modified staples: {len(spec.modified_staples)}\n\n")

        f.write(f"Tool: {spec.tool}\n")
        f.write(f"Mode: {spec.mode}\n\n")

        if spec.modified_staples:
            f.write("Modified Staples (interior modules):\n")
            f.write("-" * 40 + "\n")
            for ms in spec.modified_staples:
                f.write(f"  {ms.staple_id}: {ms.module_name}\n")
                f.write(f"    Chemistry: {ms.conjugation}\n")
                f.write(f"    Linker: {ms.linker_type} ({ms.linker_length_nm:.1f} nm)\n")
                f.write(f"    SMILES: {ms.module_smiles}\n")
                f.write(f"    Target pos: ({ms.target_position_nm[0]:.2f}, "
                        f"{ms.target_position_nm[1]:.2f}, "
                        f"{ms.target_position_nm[2]:.2f}) nm\n\n")

        f.write("\nTo run ATHENA:\n")
        f.write(f"  python -m ATHENA {spec.cage_id}.ply "
                f"--edge-type {spec.edge_type} "
                f"--scaffold {spec.scaffold}\n\n")
        f.write("To run pyDAEDALUS:\n")
        f.write(f"  python -m pyDAEDALUS --ply {spec.cage_id}.ply\n")

    files["readme"] = readme_path

    return files
