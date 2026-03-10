"""
core/dna_origami_pocket.py — Tertiary Binding via DNA Origami

The DNA wireframe is NOT a carrier for pre-made binders.
The wireframe IS the binder: it positions molecular modules such that
their combined geometry creates a binding pocket no single module achieves.

Architecture:
    InteractionGeometrySpec (Layer 2)
        → decompose_to_modules()
            → [MolecularModule, ...] each providing one interaction element
        → select_cage_geometry()
            → CagePreset with interior staple positions
        → map_modules_to_staples()
            → ModulePlacement with staple_id, linker, conjugation chemistry
        → generate_athena_spec()
            → ATHENASpec for staple sequence generation

Physics constraints:
    - DNA origami staple termini: ~2-3 nm minimum spacing (helical pitch)
    - Linker chemistry bridges sub-nm precision gap
    - Module positions must satisfy inter-module distance constraints from spec
    - Cage interior volume must accommodate target guest + modules
    - Pore size must allow guest entry but not guest escape during binding

Does NOT:
    - Run ATHENA/pyDAEDALUS (generates input specs for them)
    - Use fitted parameters against DNA binding data
    - Assume any specific cage topology a priori
"""

import math
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# MOLECULAR MODULES — sub-pocket functional elements
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MolecularModule:
    """One functional module that provides a single interaction element.

    Each module is a small molecule (1-5 heavy atoms in the functional group)
    attached to a linker that connects to a DNA staple terminus.
    The module's functional group points inward toward the pocket center.
    """
    module_id: str
    name: str
    role: str                    # "hb_donor", "hb_acceptor", "aromatic_wall",
                                 # "hydrophobic_wall", "metal_coord", "electrostatic"
    functional_group: str        # chemical identity: "catechol", "pyridine", "urea", etc.
    smiles: str                  # SMILES of the module (functional group + linker stub)
    interaction_energy_kJ: float # estimated per-module contribution

    # Spatial requirements
    functional_radius_A: float   # how far the active group extends from anchor
    required_orientation: str    # "inward", "lateral", "any"

    # Conjugation to staple
    conjugation_handle: str      # "NHS-amine", "azide-SPAAC", "thiol-maleimide"
    linker_length_nm: float      # adjustable PEG/alkyl linker length
    linker_type: str             # "PEG4", "C6-alkyl", "rigid-phenyl", etc.


# Module library: maps pocket interaction elements to molecular modules
MODULE_LIBRARY = [
    # ── H-bond acceptors (pocket provides lone pair to guest NH) ──
    MolecularModule(
        "hba_pyridine", "pyridine acceptor", "hb_acceptor",
        "pyridine", "[*]CCc1ccccn1", -6.0,
        4.5, "inward", "azide-SPAAC", 1.0, "PEG4",
    ),
    MolecularModule(
        "hba_crown_O", "crown ether oxygen", "hb_acceptor",
        "ethylene_glycol", "[*]COCCO", -3.5,
        3.0, "inward", "azide-SPAAC", 0.8, "PEG2",
    ),
    MolecularModule(
        "hba_imidazole", "imidazole acceptor", "hb_acceptor",
        "imidazole", "[*]CCc1cnc[nH]1", -5.5,
        4.5, "inward", "azide-SPAAC", 1.0, "PEG4",
    ),

    # ── H-bond donors (pocket provides NH/OH to guest C=O) ──
    MolecularModule(
        "hbd_urea", "urea donor", "hb_donor",
        "urea", "[*]CCNC(=O)N", -8.0,
        4.0, "inward", "NHS-amine", 1.0, "C6-alkyl",
    ),
    MolecularModule(
        "hbd_catechol", "catechol donor", "hb_donor",
        "catechol", "[*]CCc1ccc(O)c(O)c1", -7.5,
        5.0, "inward", "azide-SPAAC", 1.2, "PEG4",
    ),
    MolecularModule(
        "hbd_squaramide", "squaramide donor", "hb_donor",
        "squaramide", "[*]CCNC1=C(O)C(=O)C1=O", -9.0,
        4.5, "inward", "NHS-amine", 1.0, "C6-alkyl",
    ),
    MolecularModule(
        "hbd_hydroxyl", "hydroxyl donor", "hb_donor",
        "hydroxyl", "[*]CCCO", -4.0,
        3.5, "inward", "azide-SPAAC", 0.8, "PEG2",
    ),
    MolecularModule(
        "hbd_guanidinium", "guanidinium donor", "hb_donor",
        "guanidinium", "[*]CCNC(=N)N", -10.0,
        4.5, "inward", "NHS-amine", 1.0, "C6-alkyl",
    ),
    MolecularModule(
        "hbd_amidinium", "amidinium donor", "hb_donor",
        "amidinium", "[*]CCC(=N)N", -8.5,
        4.0, "inward", "NHS-amine", 1.0, "C6-alkyl",
    ),

    # ── Aromatic walls (π-stacking with guest aromatics) ──
    MolecularModule(
        "aro_naphthalene", "naphthalene wall", "aromatic_wall",
        "naphthalene", "[*]CCc1ccc2ccccc2c1", -5.0,
        5.5, "lateral", "azide-SPAAC", 1.5, "PEG4",
    ),
    MolecularModule(
        "aro_pyrene", "pyrene wall", "aromatic_wall",
        "pyrene", "[*]CCc1ccc2ccc3ccc4ccccc4c3c2c1", -7.0,
        6.0, "lateral", "azide-SPAAC", 1.5, "PEG4",
    ),
    MolecularModule(
        "aro_phenyl", "phenyl wall", "aromatic_wall",
        "phenyl", "[*]CCc1ccccc1", -3.5,
        4.5, "lateral", "azide-SPAAC", 1.0, "PEG4",
    ),

    # ── Hydrophobic walls (VdW contact with guest alkyl regions) ──
    MolecularModule(
        "hph_adamantyl", "adamantyl wall", "hydrophobic_wall",
        "adamantane", "[*]CCC12CC3CC(CC(C3)C1)C2", -4.0,
        5.0, "lateral", "azide-SPAAC", 1.0, "C6-alkyl",
    ),
    MolecularModule(
        "hph_cyclohexyl", "cyclohexyl wall", "hydrophobic_wall",
        "cyclohexane", "[*]CCC1CCCCC1", -2.5,
        4.0, "lateral", "azide-SPAAC", 1.0, "C6-alkyl",
    ),

    # ── Metal coordination modules (for metal-binding pockets) ──
    MolecularModule(
        "met_Zr_urea", "Zr-urea bidentate", "metal_coord",
        "Zr_urea", "[*]CCNC(=O)NC(=O)O", -15.0,
        4.0, "inward", "NHS-amine", 1.0, "C6-alkyl",
    ),
    MolecularModule(
        "met_hydroxamate", "hydroxamate chelator", "metal_coord",
        "hydroxamate", "[*]CCC(=O)NO", -12.0,
        3.5, "inward", "NHS-amine", 1.0, "C6-alkyl",
    ),

    # ── Charge-transfer modules (quinone-specific) ──
    MolecularModule(
        "ct_EDOT", "EDOT charge-transfer", "charge_transfer",
        "EDOT", "[*]CCC1=CSC2=C1OCCO2", -6.0,
        4.5, "inward", "azide-SPAAC", 1.2, "PEG4",
    ),
]

_MODULE_BY_ID = {m.module_id: m for m in MODULE_LIBRARY}


# ═══════════════════════════════════════════════════════════════════════════
# CAGE GEOMETRY PRESETS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class InteriorStaplePosition:
    """A staple terminus pointing into the cage interior."""
    staple_id: str
    position_nm: tuple           # (x, y, z) relative to cage center, in nm
    direction: tuple             # unit vector pointing inward
    helix_id: int                # which helix this terminus belongs to
    available: bool = True       # not yet assigned to a module


@dataclass
class CagePreset:
    """Pre-computed DNA origami cage geometry.

    Each preset has known interior volume, pore sizes, and
    pre-classified interior staple positions where modules can attach.
    """
    cage_id: str
    name: str
    polyhedron: str              # "tetrahedron", "octahedron", "icosahedron", etc.
    edge_type: str               # "DX" (double crossover) or "6HB" (6-helix bundle)
    edge_length_nm: float
    scaffold: str                # "M13mp18_7249", "p8064", etc.

    # Geometry
    interior_volume_nm3: float
    interior_volume_A3: float    # = interior_volume_nm3 * 1000
    pore_diameter_nm: float      # largest face pore
    n_faces: int
    n_edges: int
    n_vertices: int

    # Staple positions
    interior_staples: list = field(default_factory=list)  # list[InteriorStaplePosition]
    n_interior_staples: int = 0
    n_exterior_staples: int = 0

    # Practical
    folding_time_hr: float = 20.0
    cost_estimate_usd: float = 500.0
    staple_count: int = 0

    @property
    def interior_volume_compatible(self):
        """Volume available for guest + modules after module placement."""
        # Each module occupies ~0.5-1.0 nm³
        return self.interior_volume_nm3

    @property
    def min_inter_staple_nm(self):
        """Minimum distance between adjacent interior staple positions."""
        if len(self.interior_staples) < 2:
            return float("inf")
        min_d = float("inf")
        for i in range(len(self.interior_staples)):
            for j in range(i+1, len(self.interior_staples)):
                p1 = self.interior_staples[i].position_nm
                p2 = self.interior_staples[j].position_nm
                d = math.sqrt(sum((a-b)**2 for a, b in zip(p1, p2)))
                if d < min_d:
                    min_d = d
        return min_d


def _generate_interior_positions(polyhedron, edge_nm, n_per_edge=2):
    """Generate interior staple positions for a wireframe polyhedron.

    Places n_per_edge staple termini along each edge, pointing inward
    (toward cage center). DNA helices run along edges; staple termini
    at crossover points can be extended inward.
    """
    # Vertex coordinates for standard polyhedra (unit edge length)
    vertices, edges, faces = _polyhedron_topology(polyhedron)

    positions = []
    sid = 0
    for i, (v1_idx, v2_idx) in enumerate(edges):
        v1 = [c * edge_nm for c in vertices[v1_idx]]
        v2 = [c * edge_nm for c in vertices[v2_idx]]

        # Place staple points along edge
        for k in range(1, n_per_edge + 1):
            t = k / (n_per_edge + 1)
            pos = tuple(v1[d] + t * (v2[d] - v1[d]) for d in range(3))

            # Inward direction: from pos toward cage center (0,0,0)
            r = math.sqrt(sum(c**2 for c in pos))
            if r > 0.01:
                direction = tuple(-c / r for c in pos)
            else:
                direction = (0, 0, -1)

            positions.append(InteriorStaplePosition(
                staple_id=f"s{sid:03d}",
                position_nm=pos,
                direction=direction,
                helix_id=i,
            ))
            sid += 1

    return positions


def _polyhedron_topology(name):
    """Return (vertices, edges, faces) for standard polyhedra.

    Vertices are unit-edge-length coordinates.
    """
    phi = (1 + math.sqrt(5)) / 2  # golden ratio

    if name == "tetrahedron":
        # Regular tetrahedron, edge length = sqrt(8/3)
        s = 1.0 / math.sqrt(8/3)
        vertices = [
            (s, s, s), (s, -s, -s), (-s, s, -s), (-s, -s, s)
        ]
        edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        faces = 4

    elif name == "octahedron":
        s = 1.0 / math.sqrt(2)
        vertices = [
            (s,0,0),(-s,0,0),(0,s,0),(0,-s,0),(0,0,s),(0,0,-s)
        ]
        edges = [
            (0,2),(0,3),(0,4),(0,5),(1,2),(1,3),(1,4),(1,5),
            (2,4),(2,5),(3,4),(3,5)
        ]
        faces = 8

    elif name == "icosahedron":
        s = 0.5 / math.sin(2 * math.pi / 5)  # normalize to unit edge
        vertices = [
            (0, s, s*phi), (0, s, -s*phi), (0, -s, s*phi), (0, -s, -s*phi),
            (s, s*phi, 0), (-s, s*phi, 0), (s, -s*phi, 0), (-s, -s*phi, 0),
            (s*phi, 0, s), (s*phi, 0, -s), (-s*phi, 0, s), (-s*phi, 0, -s),
        ]
        edges = [
            (0,2),(0,4),(0,5),(0,8),(0,10),(1,3),(1,4),(1,5),(1,9),(1,11),
            (2,6),(2,7),(2,8),(2,10),(3,6),(3,7),(3,9),(3,11),
            (4,5),(4,8),(4,9),(5,10),(5,11),(6,7),(6,8),(6,9),
            (7,10),(7,11),(8,9),(10,11),
        ]
        faces = 20

    elif name == "truncated_octahedron":
        # Truncated octahedron: 24 vertices, 36 edges, 14 faces
        # Vertices at permutations of (0, ±1, ±2) (normalized)
        s = 0.5  # scale to ~unit edge
        raw = []
        for signs in [(1,1),(1,-1),(-1,1),(-1,-1)]:
            for perm in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]:
                v = [0.0, 0.0, 0.0]
                vals = [0, signs[0], signs[1] * 2]
                v[perm[0]] = vals[0] * s
                v[perm[1]] = vals[1] * s
                v[perm[2]] = vals[2] * s
                t = tuple(v)
                if t not in raw:
                    raw.append(t)
        vertices = raw[:24]
        # Edges: connect vertices that are distance sqrt(2)*s apart
        edges = []
        target_d = math.sqrt(2) * s
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                d = math.sqrt(sum((a-b)**2 for a,b in zip(vertices[i], vertices[j])))
                if abs(d - target_d) < 0.01 * s:
                    edges.append((i, j))
        faces = 14

    else:
        # Default: octahedron
        return _polyhedron_topology("octahedron")

    return vertices, edges, faces


# Pre-built cage presets
CAGE_PRESETS = []


def _build_cage_presets():
    """Build standard cage presets with interior staple positions."""
    global CAGE_PRESETS

    configs = [
        # (id, name, polyhedron, edge_type, edge_nm, scaffold, n_per_edge)
        ("tet_20_DX", "Tetrahedron 20nm DX", "tetrahedron", "DX",
         20.0, "M13mp18_7249", 2),
        ("oct_30_DX", "Octahedron 30nm DX", "octahedron", "DX",
         30.0, "M13mp18_7249", 3),
        ("oct_40_6HB", "Octahedron 40nm 6HB", "octahedron", "6HB",
         40.0, "p8064", 4),
        ("ico_40_DX", "Icosahedron 40nm DX", "icosahedron", "DX",
         40.0, "p8064", 3),
        ("trunc_oct_35_6HB", "Truncated Octahedron 35nm 6HB",
         "truncated_octahedron", "6HB", 35.0, "p8064", 3),
    ]

    for cid, name, poly, etype, edge_nm, scaffold, npe in configs:
        interior = _generate_interior_positions(poly, edge_nm, n_per_edge=npe)
        _, edges, n_faces = _polyhedron_topology(poly)
        n_verts = len(set(v for e in edges for v in e))

        # Interior volume estimate from edge length
        # Tetrahedron: V = edge³/(6√2), Octahedron: V = edge³√2/3,
        # Icosahedron: V = 5(3+√5)/12 * edge³
        if poly == "tetrahedron":
            vol_nm3 = edge_nm**3 / (6 * math.sqrt(2))
        elif poly == "octahedron":
            vol_nm3 = edge_nm**3 * math.sqrt(2) / 3
        elif poly == "icosahedron":
            vol_nm3 = 5 * (3 + math.sqrt(5)) / 12 * edge_nm**3
        elif poly == "truncated_octahedron":
            vol_nm3 = 8 * math.sqrt(2) * edge_nm**3
        else:
            vol_nm3 = edge_nm**3

        # Pore diameter: inscribed circle of largest face
        # Triangle face: pore_d ≈ edge / √3 (for DX, minus helix diameter ~2nm)
        helix_d = 2.0 if etype == "DX" else 4.0  # nm
        if poly in ("tetrahedron", "octahedron", "icosahedron"):
            pore_d = edge_nm / math.sqrt(3) - helix_d
        else:
            pore_d = edge_nm / 2 - helix_d  # square/hex faces are larger

        # Staple count estimate
        bp_per_edge = edge_nm * 10 / 0.34  # 0.34 nm per bp
        total_bp = bp_per_edge * len(edges)
        staple_count = int(total_bp / 32)  # ~32 bp per staple

        preset = CagePreset(
            cage_id=cid,
            name=name,
            polyhedron=poly,
            edge_type=etype,
            edge_length_nm=edge_nm,
            scaffold=scaffold,
            interior_volume_nm3=vol_nm3,
            interior_volume_A3=vol_nm3 * 1000,
            pore_diameter_nm=max(0, pore_d),
            n_faces=n_faces,
            n_edges=len(edges),
            n_vertices=n_verts,
            interior_staples=interior,
            n_interior_staples=len(interior),
            n_exterior_staples=len(interior),  # ~same count outside
            staple_count=staple_count,
        )
        CAGE_PRESETS.append(preset)

_build_cage_presets()
_CAGE_BY_ID = {c.cage_id: c for c in CAGE_PRESETS}


# ═══════════════════════════════════════════════════════════════════════════
# MODULE DECOMPOSITION: InteractionGeometrySpec → MolecularModule list
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ModuleRequirement:
    """One required module derived from the pocket spec."""
    spec_element_idx: int        # index into spec.donor_positions or hydrophobic_surfaces
    role: str                    # "hb_donor", "hb_acceptor", "aromatic_wall", "hydrophobic_wall"
    position_A: tuple            # required position (from spec)
    position_nm: tuple           # same in nm
    candidates: list = field(default_factory=list)  # list[MolecularModule]
    selected: object = None      # chosen MolecularModule


def decompose_to_modules(spec) -> list:
    """Convert InteractionGeometrySpec donor positions + surfaces into module requirements.

    Each DonorPosition with charge < 0 (pocket acceptor) → needs hb_acceptor module
    Each DonorPosition with charge > 0 (pocket donor) → needs hb_donor module
    Each HydrophobicSurface → needs aromatic_wall or hydrophobic_wall module

    Returns list[ModuleRequirement].
    """
    requirements = []

    for i, dp in enumerate(spec.donor_positions):
        if dp.charge_state < 0:
            role = "hb_acceptor"
        else:
            role = "hb_donor"

        pos_A = dp.position_vector_A
        pos_nm = tuple(c / 10.0 for c in pos_A)

        candidates = [m for m in MODULE_LIBRARY if m.role == role]
        req = ModuleRequirement(
            spec_element_idx=i,
            role=role,
            position_A=pos_A,
            position_nm=pos_nm,
            candidates=candidates,
        )
        # Select best candidate (highest |interaction_energy|)
        if candidates:
            req.selected = max(candidates, key=lambda m: abs(m.interaction_energy_kJ))
        requirements.append(req)

    # Hydrophobic/aromatic surfaces — take up to 4 most important
    for i, hs in enumerate(spec.hydrophobic_surfaces[:4]):
        pos_A = hs.center_A
        pos_nm = tuple(c / 10.0 for c in pos_A)

        # Large area → aromatic wall; small area → hydrophobic wall
        if hs.area_A2 >= 20.0:
            role = "aromatic_wall"
        else:
            role = "hydrophobic_wall"

        candidates = [m for m in MODULE_LIBRARY if m.role == role]
        req = ModuleRequirement(
            spec_element_idx=len(spec.donor_positions) + i,
            role=role,
            position_A=pos_A,
            position_nm=pos_nm,
            candidates=candidates,
        )
        if candidates:
            req.selected = max(candidates, key=lambda m: abs(m.interaction_energy_kJ))
        requirements.append(req)

    return requirements


# ═══════════════════════════════════════════════════════════════════════════
# CAGE SELECTION: pick geometry that satisfies spatial constraints
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CageScore:
    """Evaluation of a cage geometry for a set of module requirements."""
    cage: CagePreset
    score: float = 0.0
    volume_ok: bool = True
    pore_ok: bool = True
    n_modules_placeable: int = 0
    placement_rmsd_nm: float = 0.0
    notes: str = ""


def select_cage_geometry(
    requirements: list,
    guest_volume_A3: float,
    guest_max_dim_A: float = 20.0,
) -> list:
    """Select best cage geometry for module placement.

    Criteria:
    1. Interior volume must fit guest + modules
    2. Pore must allow guest entry (pore_d > guest_min_dim)
    3. Interior staple positions must cover module position requirements
    4. Smaller cages preferred (tighter pocket = better selectivity)

    Returns list[CageScore] sorted best-first.
    """
    guest_vol_nm3 = guest_volume_A3 / 1000.0
    module_vol_nm3 = len(requirements) * 0.8  # ~0.8 nm³ per module
    required_vol = (guest_vol_nm3 + module_vol_nm3) * 2.0  # 2× for access

    # Guest must fit through pore
    guest_pore_req_nm = guest_max_dim_A / 10.0 * 0.6  # need ~60% of max dim

    scores = []
    for cage in CAGE_PRESETS:
        cs = CageScore(cage=cage)

        # Volume check
        if cage.interior_volume_nm3 < required_vol:
            cs.volume_ok = False
            cs.notes += "too_small; "

        # Pore check
        if cage.pore_diameter_nm < guest_pore_req_nm:
            cs.pore_ok = False
            cs.notes += f"pore {cage.pore_diameter_nm:.1f}<{guest_pore_req_nm:.1f}nm; "

        # Module placement check: how many requirements can be satisfied
        placements, angular_rmsd = _try_place_modules(requirements, cage)
        cs.n_modules_placeable = len(placements)
        cs.placement_rmsd_nm = angular_rmsd  # actually angular RMSD in radians

        # Score for tertiary binding:
        # - All modules must be placeable
        # - Good angular match (modules point the right way)
        # - Smaller cage preferred (shorter linkers = tighter pocket)
        # - Pore must admit guest
        # Volume is intentionally much larger than guest — the cage
        # provides the scaffold, not the cavity encapsulation.
        if not cs.volume_ok:
            cs.score = -20.0
        elif not cs.pore_ok:
            cs.score = -10.0
        elif cs.n_modules_placeable < len(requirements):
            frac = cs.n_modules_placeable / max(len(requirements), 1)
            cs.score = frac * 3.0 - angular_rmsd
        else:
            # Full placement — score by angular quality + prefer smaller
            angular_quality = max(0, 3.0 - angular_rmsd * 3.0)  # 0-3 points
            # Smaller cage = shorter linkers = more preorganized
            size_bonus = max(0, 2.0 - cage.edge_length_nm / 30.0)  # 0-2 points
            cs.score = angular_quality + size_bonus

        scores.append(cs)

    scores.sort(key=lambda c: c.score, reverse=True)
    return scores


def _try_place_modules(requirements, cage):
    """Try to assign each module requirement to a cage staple position.

    Uses angular matching: module positions (from pocket spec) define
    DIRECTIONS from the pocket center. Staple positions define POSITIONS
    on the cage interior wall. We match by angular proximity — the staple
    whose inward direction best aligns with the module's required direction.

    Returns (placements, avg_angular_error_rad).
    """
    available = list(cage.interior_staples)
    placements = []
    total_angle_sq = 0.0

    for req in requirements:
        if not available:
            break

        # Module direction: unit vector from origin toward required position
        pos = req.position_nm
        r = math.sqrt(sum(c**2 for c in pos))
        if r < 0.001:
            req_dir = (0, 0, 1)
        else:
            req_dir = tuple(c / r for c in pos)

        # Find staple whose position (from center) best matches this direction
        best_idx = -1
        best_angle = float("inf")
        for idx, staple in enumerate(available):
            sp = staple.position_nm
            sr = math.sqrt(sum(c**2 for c in sp))
            if sr < 0.001:
                continue
            s_dir = tuple(c / sr for c in sp)
            # Cosine similarity
            dot = sum(a * b for a, b in zip(req_dir, s_dir))
            dot = max(-1.0, min(1.0, dot))
            angle = math.acos(dot)
            if angle < best_angle:
                best_angle = angle
                best_idx = idx

        if best_idx >= 0:
            placements.append((req, available[best_idx], best_angle))
            total_angle_sq += best_angle ** 2
            available.pop(best_idx)

    n = len(placements)
    rmsd = math.sqrt(total_angle_sq / n) if n > 0 else float("inf")
    return placements, rmsd


# ═══════════════════════════════════════════════════════════════════════════
# MODULE-TO-STAPLE MAPPING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ModulePlacement:
    """Final assignment of a module to a staple position."""
    module: MolecularModule
    staple_id: str
    staple_position_nm: tuple
    required_position_nm: tuple
    offset_nm: float             # distance between staple and required position
    linker_length_nm: float      # adjusted linker to bridge the offset
    conjugation: str             # conjugation chemistry
    notes: str = ""


def map_modules_to_staples(
    requirements: list,
    cage: CagePreset,
) -> list:
    """Assign modules to cage interior staple positions.

    For each module, adjusts linker length to bridge the gap between
    staple terminus position and required functional group position.

    Returns list[ModulePlacement].
    """
    available = list(cage.interior_staples)
    placements = []

    for req in requirements:
        if not available or req.selected is None:
            continue

        # Module direction from origin
        pos = req.position_nm
        r = math.sqrt(sum(c**2 for c in pos))
        if r < 0.001:
            req_dir = (0, 0, 1)
        else:
            req_dir = tuple(c / r for c in pos)

        # Find staple whose position direction best matches
        best_idx = -1
        best_angle = float("inf")
        for idx, staple in enumerate(available):
            sp = staple.position_nm
            sr = math.sqrt(sum(c**2 for c in sp))
            if sr < 0.001:
                continue
            s_dir = tuple(c / sr for c in sp)
            dot = sum(a * b for a, b in zip(req_dir, s_dir))
            dot = max(-1.0, min(1.0, dot))
            angle = math.acos(dot)
            if angle < best_angle:
                best_angle = angle
                best_idx = idx

        if best_idx < 0:
            continue

        staple = available.pop(best_idx)
        module = req.selected

        # Linker length: from staple on cage wall to pocket interior
        # Staple is on the wall; module must reach toward cage center
        # Distance from staple to center ≈ cage radius, module needs to
        # reach ~halfway in (where the guest sits)
        staple_r = math.sqrt(sum(c**2 for c in staple.position_nm))
        # Module reach needed: from wall to ~1/3 of radius (pocket region)
        reach_needed_nm = staple_r * 0.3
        needed_linker = max(0.5, reach_needed_nm - module.functional_radius_A / 10.0)

        # Cap linker at practical maximum (PEG12 ≈ 5 nm)
        if needed_linker > 5.0:
            notes = f"linker {needed_linker:.1f}nm exceeds PEG12 max (5nm)"
            needed_linker = 5.0
        else:
            notes = ""

        placements.append(ModulePlacement(
            module=module,
            staple_id=staple.staple_id,
            staple_position_nm=staple.position_nm,
            required_position_nm=req.position_nm,
            offset_nm=best_angle,  # angular error in radians
            linker_length_nm=round(needed_linker, 2),
            conjugation=module.conjugation_handle,
            notes=notes,
        ))

    return placements


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API: POCKET SPEC → DNA ORIGAMI TERTIARY BINDING DESIGN
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DNAOrigamiPocketDesign:
    """Complete tertiary binding design for a DNA origami cage."""
    guest_name: str
    guest_smiles: str

    # Module decomposition
    modules: list = field(default_factory=list)          # list[ModuleRequirement]
    n_modules: int = 0

    # Cage selection
    cage: object = None                                  # CagePreset
    cage_scores: list = field(default_factory=list)      # list[CageScore]

    # Module placement
    placements: list = field(default_factory=list)       # list[ModulePlacement]
    placement_rmsd_nm: float = 0.0

    # ATHENA spec
    athena_spec: object = None                           # ATHENASpec

    # Cost/feasibility
    estimated_cost_usd: float = 0.0
    n_modified_staples: int = 0
    conjugation_chemistries: list = field(default_factory=list)
    feasibility_grade: str = ""  # "excellent", "good", "challenging", "infeasible"

    # Predicted binding
    total_interaction_energy_kJ: float = 0.0
    estimated_log_Ka: float = 0.0


def design_dna_origami_pocket(
    spec,
    guest_smiles: str,
    guest_name: str = "",
    guest_volume_A3: float = 0.0,
    guest_max_dim_A: float = 20.0,
    max_modules: int = 8,
) -> DNAOrigamiPocketDesign:
    """Design a DNA origami tertiary binding pocket for a guest.

    Args:
        spec: InteractionGeometrySpec from guest_to_pocket_spec()
        guest_smiles: target guest SMILES
        guest_name: display name
        guest_volume_A3: guest molecular volume (for cage sizing)
        guest_max_dim_A: guest max dimension (for pore sizing)
        max_modules: maximum number of functional modules to place

    Returns:
        DNAOrigamiPocketDesign with cage, module placements, ATHENA spec.
    """
    from core.athena_interface import generate_athena_spec

    result = DNAOrigamiPocketDesign(
        guest_name=guest_name or guest_smiles[:40],
        guest_smiles=guest_smiles,
    )

    # Step 1: Decompose pocket spec into module requirements
    modules = decompose_to_modules(spec)
    if len(modules) > max_modules:
        # Keep most energetically important
        modules.sort(key=lambda m: abs(m.selected.interaction_energy_kJ)
                      if m.selected else 0, reverse=True)
        modules = modules[:max_modules]
    result.modules = modules
    result.n_modules = len(modules)

    if not modules:
        result.feasibility_grade = "infeasible"
        return result

    # Step 2: Select cage geometry
    cage_scores = select_cage_geometry(
        modules, guest_volume_A3, guest_max_dim_A,
    )
    result.cage_scores = cage_scores

    # Pick best feasible cage
    feasible = [cs for cs in cage_scores if cs.score > 0]
    if not feasible:
        result.feasibility_grade = "infeasible"
        result.cage = cage_scores[0].cage if cage_scores else None
        return result

    best = feasible[0]
    result.cage = best.cage

    # Step 3: Map modules to staple positions
    placements = map_modules_to_staples(modules, best.cage)
    result.placements = placements
    result.n_modified_staples = len(placements)

    # Placement quality
    if placements:
        result.placement_rmsd_nm = math.sqrt(
            sum(p.offset_nm**2 for p in placements) / len(placements)
        )

    # Step 4: Summarize
    result.conjugation_chemistries = list(set(p.conjugation for p in placements))

    # Cost: base cage + modified staples
    base_cost = best.cage.cost_estimate_usd
    mod_cost = len(placements) * 50  # ~$50 per modified staple with linker+module
    result.estimated_cost_usd = base_cost + mod_cost

    # Predicted binding: sum of module interaction energies (additive approximation)
    total_E = sum(
        p.module.interaction_energy_kJ for p in placements
    )
    result.total_interaction_energy_kJ = total_E
    result.estimated_log_Ka = -total_E / 5.71 if total_E != 0 else 0

    # Feasibility grade (placement_rmsd is now angular error in radians)
    # < 0.5 rad (~30°) = good angular match
    # < 1.0 rad (~60°) = challenging but workable with linker adjustment
    if result.placement_rmsd_nm < 0.5 and len(placements) >= len(modules):
        result.feasibility_grade = "good"
    elif result.placement_rmsd_nm < 1.0 and len(placements) >= len(modules) * 0.7:
        result.feasibility_grade = "challenging"
    else:
        result.feasibility_grade = "speculative"

    # Step 5: Generate ATHENA spec
    result.athena_spec = generate_athena_spec(best.cage, placements)

    return result
