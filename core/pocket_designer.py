"""
core/pocket_designer.py — 3D Synthetic Pocket Design Engine

Physics-first design of selective binding pockets from cage topologies.
No biological training bias. All scoring from calibrated NIST/HG parameters.

Pipeline:
  1. Target spec (metal + interferents + pH + conditions)
  2. HSAB-guided donor selection (which donor atoms does physics want?)
  3. Cage topology enumeration (Platonic/Archimedean solids, tubes, prisms)
  4. Donor placement on interior geometry (vertex/face/edge positions)
  5. Score each design: affinity + selectivity via unified scorer
  6. Rank and output CageDesignSpec (ATHENA-ready)

Usage:
    from core.pocket_designer import design_pocket, print_pocket_designs
    results = design_pocket("Hg2+", interferents=["Zn2+", "Pb2+"], pH=5.0)
    print_pocket_designs(results)
"""

import sys
import os
import math
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.scorer_frozen import (
    predict_log_k, SUBTYPE_EXCHANGE, METAL_DB,
    DONOR_SOFTNESS, DONOR_PKA,
)


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CageTopology:
    """A polyhedral cage geometry."""
    name: str
    vertices: np.ndarray          # Nx3 vertex coordinates (unit edge)
    edges: list                   # list of (i,j) vertex index pairs
    faces: list                   # list of [i,j,k,...] vertex index lists
    n_vertices: int
    n_edges: int
    n_faces: int
    inscribed_radius: float       # inscribed sphere radius (unit edge)
    circumscribed_radius: float   # circumscribed sphere radius (unit edge)
    cavity_volume_factor: float   # V_cavity / edge_length^3

    def at_edge_length(self, edge_nm):
        """Return scaled vertices and cavity properties at given edge length."""
        scale = edge_nm / self._unit_edge_length()
        return {
            "vertices_nm": self.vertices * scale,
            "edge_nm": edge_nm,
            "cavity_radius_nm": self.inscribed_radius * scale,
            "cavity_volume_nm3": self.cavity_volume_factor * edge_nm**3,
            "cavity_volume_A3": self.cavity_volume_factor * edge_nm**3 * 1000,
        }

    def _unit_edge_length(self):
        i, j = self.edges[0]
        return np.linalg.norm(self.vertices[i] - self.vertices[j])


@dataclass
class DonorPlacement:
    """A set of donor atoms placed inside a cage."""
    donor_subtypes: list          # e.g. ["S_thiolate", "S_thiolate", "S_thiolate"]
    positions_nm: np.ndarray      # Nx3 positions inside cavity
    pointing_inward: bool = True  # donors face cavity center
    linker_chemistry: str = ""    # how donors attach to cage (e.g. "thiol-maleimide")
    description: str = ""


@dataclass
class CageDesignSpec:
    """Complete pocket design — ATHENA-consumable."""
    # Topology
    topology_name: str
    edge_length_nm: float
    n_vertices: int
    n_edges: int
    vertex_coords_nm: np.ndarray
    edge_list: list
    cavity_radius_nm: float
    cavity_volume_A3: float

    # Donor configuration
    donor_subtypes: list
    donor_positions_nm: np.ndarray
    donor_count: int
    denticity: int
    is_macrocyclic: bool
    linker_chemistry: str

    # Scoring
    target_metal: str
    target_log_ka: float
    target_dg_kj: float
    interferent_scores: dict = field(default_factory=dict)
    selectivity_gaps: dict = field(default_factory=dict)
    min_selectivity_gap: float = 0.0
    worst_interferent: str = ""
    selectivity_grade: str = ""

    # Design metadata
    design_id: str = ""
    notes: str = ""
    pH: float = 7.0
    rank: int = 0

    def to_athena_input(self):
        """Export geometry for ATHENA consumption."""
        return {
            "topology": self.topology_name,
            "edge_length_nm": self.edge_length_nm,
            "vertices": self.vertex_coords_nm.tolist(),
            "edges": self.edge_list,
            "donor_positions": self.donor_positions_nm.tolist(),
            "donor_types": self.donor_subtypes,
            "linker": self.linker_chemistry,
            "target": self.target_metal,
            "predicted_log_ka": self.target_log_ka,
        }


# ═══════════════════════════════════════════════════════════════════════════
# CAGE TOPOLOGY LIBRARY
# ═══════════════════════════════════════════════════════════════════════════

def _make_topologies():
    """Build library of polyhedral cage topologies."""
    phi = (1 + math.sqrt(5)) / 2
    topos = {}

    # ── Tetrahedron (M4L6) ──
    v = np.array([[1,1,1],[-1,-1,1],[-1,1,-1],[1,-1,-1]], dtype=float)
    v = v / np.linalg.norm(v[0] - v[1])  # normalize to unit edge
    edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    faces = [[0,1,2],[0,1,3],[0,2,3],[1,2,3]]
    r_in = 1 / (2 * math.sqrt(6))   # inscribed radius for unit edge
    r_out = math.sqrt(6) / 4
    v_cav = math.sqrt(2) / 12        # volume for unit edge
    topos["tetrahedron"] = CageTopology(
        "tetrahedron", v, edges, faces, 4, 6, 4, r_in, r_out, v_cav)

    # ── Octahedron (M6L12) ──
    v = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=float)
    v = v / math.sqrt(2)  # unit edge
    edges = [(0,2),(0,3),(0,4),(0,5),(1,2),(1,3),(1,4),(1,5),
             (2,4),(2,5),(3,4),(3,5)]
    faces = [[0,2,4],[0,4,3],[0,3,5],[0,5,2],[1,2,4],[1,4,3],[1,3,5],[1,5,2]]
    r_in = 1 / math.sqrt(3)
    r_out = 1 / math.sqrt(2)
    v_cav = math.sqrt(2) / 3
    topos["octahedron"] = CageTopology(
        "octahedron", v, edges, faces, 6, 12, 8, r_in, r_out, v_cav)

    # ── Cube (M8L12) ──
    v = np.array([[i,j,k] for i in [-1,1] for j in [-1,1] for k in [-1,1]],
                 dtype=float) / 2  # unit edge
    edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    faces = [[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]
    r_in = 0.5
    r_out = math.sqrt(3)/2
    v_cav = 1.0
    topos["cube"] = CageTopology(
        "cube", v, edges, faces, 8, 12, 6, r_in, r_out, v_cav)

    # ── Truncated tetrahedron (M12L18) ──
    # Archimedean solid — good for larger cavities
    # Approximate: scale factor from tetrahedron
    r_in_tt = 0.959  # inscribed radius for unit edge
    r_out_tt = 1.470
    v_cav_tt = 2.711
    # Vertices of truncated tetrahedron (unit edge, standard coords)
    tt_v = np.array([
        [3,1,1],[1,3,1],[1,1,3],[-3,-1,1],[-1,-3,1],[-1,-1,3],
        [-3,1,-1],[-1,3,-1],[-1,1,-3],[3,-1,-1],[1,-3,-1],[1,-1,-3]
    ], dtype=float) / (2 * math.sqrt(2))
    # Edges: each pair at distance 1.0 (unit edge)
    tt_edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(tt_v[i] - tt_v[j])
            if abs(d - 1.0) < 0.01:
                tt_edges.append((i, j))
    topos["truncated_tetrahedron"] = CageTopology(
        "truncated_tetrahedron", tt_v, tt_edges, [], 12, len(tt_edges), 8,
        r_in_tt, r_out_tt, v_cav_tt)

    # ── Cuboctahedron (M12L24) ──
    co_v = np.array([
        [1,1,0],[1,-1,0],[-1,1,0],[-1,-1,0],
        [1,0,1],[1,0,-1],[-1,0,1],[-1,0,-1],
        [0,1,1],[0,1,-1],[0,-1,1],[0,-1,-1]
    ], dtype=float) / math.sqrt(2)
    co_edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(co_v[i] - co_v[j])
            if abs(d - 1.0) < 0.05:
                co_edges.append((i, j))
    r_in_co = 1.0
    r_out_co = 1.0
    v_cav_co = 5 * math.sqrt(2) / 3
    topos["cuboctahedron"] = CageTopology(
        "cuboctahedron", co_v, co_edges, [], 12, len(co_edges), 14,
        r_in_co, r_out_co, v_cav_co)

    # ── Trigonal prism (M6L9) ──
    # Common in Nitschke-type cages
    h = math.sqrt(2/3)
    tp_v = np.array([
        [1, 0, 0], [-0.5, math.sqrt(3)/2, 0], [-0.5, -math.sqrt(3)/2, 0],
        [1, 0, h], [-0.5, math.sqrt(3)/2, h], [-0.5, -math.sqrt(3)/2, h],
    ], dtype=float)
    tp_edges = [(0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(0,3),(1,4),(2,5)]
    r_in_tp = 0.289
    v_cav_tp = math.sqrt(3)/4 * h
    topos["trigonal_prism"] = CageTopology(
        "trigonal_prism", tp_v, tp_edges, [], 6, 9, 5, r_in_tp, 0.6, v_cav_tp)

    return topos


CAGE_TOPOLOGIES = _make_topologies()


# ═══════════════════════════════════════════════════════════════════════════
# DONOR SELECTION (HSAB-GUIDED)
# ═══════════════════════════════════════════════════════════════════════════

# Donor heads with attachment chemistry
DONOR_HEADS = {
    # Hard O-donors
    "O_carboxylate":  {"linker": "amide bond",    "head_smiles": "CC(=O)[O-]",
                       "description": "Carboxylate arm"},
    "O_phenolate":    {"linker": "ether linkage",  "head_smiles": "[O-]c1ccccc1",
                       "description": "Phenolate arm"},
    "O_hydroxamate":  {"linker": "amide bond",    "head_smiles": "CC(=O)NO",
                       "description": "Hydroxamate arm"},
    "O_catecholate":  {"linker": "ether linkage",  "head_smiles": "[O-]c1ccccc1[O-]",
                       "description": "Catechol arm"},
    "O_phosphate":    {"linker": "phosphoester",   "head_smiles": "OP(=O)([O-])[O-]",
                       "description": "Phosphonate arm"},
    # Borderline N-donors
    "N_amine":        {"linker": "alkyl chain",    "head_smiles": "NCC",
                       "description": "Amine arm"},
    "N_pyridine":     {"linker": "direct bond",    "head_smiles": "c1ccncc1",
                       "description": "Pyridine arm"},
    "N_imidazole":    {"linker": "alkyl chain",    "head_smiles": "c1c[nH]cn1",
                       "description": "Imidazole arm"},
    "N_imine":        {"linker": "Schiff base",    "head_smiles": "C=NC",
                       "description": "Imine/Schiff base arm"},
    # Soft S-donors
    "S_thiolate":     {"linker": "thiol-maleimide", "head_smiles": "[S-]CC",
                       "description": "Thiolate arm"},
    "S_thioether":    {"linker": "thioether bond",  "head_smiles": "CSCC",
                       "description": "Thioether arm"},
    "S_dithiocarbamate": {"linker": "CS2 addition", "head_smiles": "SC(=S)NCC",
                          "description": "Dithiocarbamate arm"},
    # Soft P-donors
    "P_phosphine":    {"linker": "P-C bond",       "head_smiles": "P(c1ccccc1)c1ccccc1",
                       "description": "Phosphine arm"},
}


def _metal_softness(metal_formula):
    """Return metal softness (0=hard, 1=soft) from METAL_DB."""
    props = METAL_DB.get(metal_formula)
    if props:
        return getattr(props, 'hsab_softness', 0.3)
    return 0.3


def select_donors_for_target(target_metal, n_donors=4, diversity=True):
    """HSAB-guided donor selection: match metal softness to donor softness.

    Returns list of donor set options, ranked by predicted exchange energy.
    """
    metal_soft = _metal_softness(target_metal)

    # Score each donor type by HSAB match
    scored_donors = []
    for subtype, props in DONOR_HEADS.items():
        donor_soft = DONOR_SOFTNESS.get(subtype, 0.3)
        # HSAB match: penalty for mismatch
        hsab_match = 1.0 - abs(metal_soft - donor_soft)
        exchange = SUBTYPE_EXCHANGE.get(subtype, -5.0)
        # Combined score: exchange energy × HSAB match
        score = -exchange * hsab_match  # more negative exchange + better match = higher score
        scored_donors.append((subtype, score, exchange, donor_soft))

    scored_donors.sort(key=lambda x: -x[1])

    # Generate donor sets
    donor_sets = []

    # Option 1: Best single donor type × n
    best = scored_donors[0][0]
    donor_sets.append([best] * n_donors)

    if diversity:
        # Option 2: Top 2 donors mixed
        if len(scored_donors) >= 2:
            d1, d2 = scored_donors[0][0], scored_donors[1][0]
            donor_sets.append([d1] * (n_donors // 2) + [d2] * (n_donors - n_donors // 2))

        # Option 3: Hard/soft mixed (heterogeneous pocket)
        hard = [d for d in scored_donors if DONOR_SOFTNESS.get(d[0], 0) < 0.3]
        soft = [d for d in scored_donors if DONOR_SOFTNESS.get(d[0], 0) >= 0.5]
        if hard and soft:
            donor_sets.append([hard[0][0]] * (n_donors // 2) + [soft[0][0]] * (n_donors - n_donors // 2))

        # Option 4: All-N (amine cage)
        donor_sets.append(["N_amine"] * n_donors)

        # Option 5: All-S (soft cage) for soft metals
        if metal_soft > 0.4:
            donor_sets.append(["S_thiolate"] * n_donors)

        # Option 6: Pyridine cage (common in Nitschke chemistry)
        donor_sets.append(["N_pyridine"] * n_donors)

    return donor_sets, scored_donors


# ═══════════════════════════════════════════════════════════════════════════
# DONOR PLACEMENT IN CAGE
# ═══════════════════════════════════════════════════════════════════════════

def place_donors_on_faces(topo, geom, n_donors):
    """Place donors at face centroids, pointing inward."""
    if not topo.faces or n_donors > len(topo.faces):
        return place_donors_on_vertices(topo, geom, n_donors)

    verts = geom["vertices_nm"]
    positions = []
    for face in topo.faces[:n_donors]:
        centroid = np.mean(verts[face], axis=0)
        # Pull slightly toward center (donors point inward)
        center = np.mean(verts, axis=0)
        direction = center - centroid
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        positions.append(centroid + 0.2 * direction)

    return np.array(positions)


def place_donors_on_vertices(topo, geom, n_donors):
    """Place donors at vertex positions."""
    verts = geom["vertices_nm"]
    n_avail = min(n_donors, len(verts))
    return verts[:n_avail].copy()


def place_donors_on_edges(topo, geom, n_donors):
    """Place donors at edge midpoints."""
    verts = geom["vertices_nm"]
    positions = []
    for i, j in topo.edges[:n_donors]:
        mid = (verts[i] + verts[j]) / 2
        center = np.mean(verts, axis=0)
        direction = center - mid
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        positions.append(mid + 0.15 * direction)
    return np.array(positions)


# ═══════════════════════════════════════════════════════════════════════════
# SCORING ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def score_cage_design(target_metal, donor_subtypes, cavity_radius_nm,
                       pH=7.0, is_macrocyclic=True):
    """Score a cage design using the calibrated metal scorer.

    The cage acts as a macrocyclic host with donors pointing inward.
    Uses scorer_frozen.predict_log_k with macrocyclic=True.
    """
    try:
        log_k = predict_log_k(
            target_metal,
            donor_subtypes,
            chelate_rings=0,         # macrocyclic convention
            ring_sizes=None,
            pH=pH,
            is_macrocyclic=True,
            cavity_radius_nm=cavity_radius_nm,
            n_ligand_molecules=1,
            temperature_K=298.15,
        )
        return log_k
    except (ValueError, KeyError):
        return None


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DESIGN ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def _grade(gap):
    if gap >= 5.0: return "A"
    elif gap >= 3.0: return "B"
    elif gap >= 1.0: return "C"
    elif gap >= 0.0: return "D"
    return "F"


# Edge lengths to explore (nm) — spanning DNA origami to small-molecule cages
EDGE_LENGTHS_NM = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]

# Number of donors to try
DONOR_COUNTS = [3, 4, 6]


def design_pocket(target_metal, interferents=None, pH=7.0,
                   edge_lengths=None, donor_counts=None,
                   topologies=None, max_results=20):
    """Design synthetic binding pockets for a target.

    Args:
        target_metal: e.g. "Hg2+", "Pb2+", "Cu2+"
        interferents: list of competing metals for selectivity
        pH: operating pH
        edge_lengths: list of edge lengths in nm to explore
        donor_counts: list of donor counts to try
        topologies: list of topology names (default: all)
        max_results: how many designs to return

    Returns:
        list of CageDesignSpec, ranked by selectivity (if interferents)
        or by affinity (if no interferents)
    """
    t0 = time.time()
    if interferents is None:
        interferents = []
    if edge_lengths is None:
        edge_lengths = EDGE_LENGTHS_NM
    if donor_counts is None:
        donor_counts = DONOR_COUNTS
    if topologies is None:
        topologies = list(CAGE_TOPOLOGIES.keys())

    # Step 1: HSAB-guided donor selection
    all_donor_sets = set()
    for nd in donor_counts:
        sets, _ = select_donors_for_target(target_metal, n_donors=nd)
        for ds in sets:
            all_donor_sets.add(tuple(sorted(ds)))

    # Step 2: Enumerate cage × edge × donors
    designs = []
    design_id = 0

    for topo_name in topologies:
        topo = CAGE_TOPOLOGIES.get(topo_name)
        if topo is None:
            continue

        for edge_nm in edge_lengths:
            geom = topo.at_edge_length(edge_nm)
            cavity_r = geom["cavity_radius_nm"]
            cavity_vol = geom["cavity_volume_A3"]

            # Skip if cavity too small for hydrated metal ion
            if cavity_r < 0.1:
                continue

            for donor_set in all_donor_sets:
                donors = list(donor_set)
                nd = len(donors)

                # Place donors
                if nd <= topo.n_faces and topo.faces:
                    positions = place_donors_on_faces(topo, geom, nd)
                elif nd <= topo.n_vertices:
                    positions = place_donors_on_vertices(topo, geom, nd)
                else:
                    positions = place_donors_on_edges(topo, geom, nd)

                if positions is None or len(positions) < nd:
                    continue

                # Score target
                log_ka = score_cage_design(
                    target_metal, donors, cavity_r, pH=pH)
                if log_ka is None:
                    continue

                LN10_RT = 5.7087
                dg = -log_ka * LN10_RT

                # Score interferents
                intf_scores = {}
                sel_gaps = {}
                for intf in interferents:
                    intf_lk = score_cage_design(
                        intf, donors, cavity_r, pH=pH)
                    if intf_lk is not None:
                        intf_scores[intf] = intf_lk
                        sel_gaps[intf] = log_ka - intf_lk

                min_gap = min(sel_gaps.values()) if sel_gaps else log_ka
                worst = min(sel_gaps, key=sel_gaps.get) if sel_gaps else ""

                # Determine linker chemistry from dominant donor
                from collections import Counter
                dominant = Counter(donors).most_common(1)[0][0]
                linker = DONOR_HEADS.get(dominant, {}).get("linker", "unknown")

                design_id += 1
                spec = CageDesignSpec(
                    topology_name=topo_name,
                    edge_length_nm=edge_nm,
                    n_vertices=topo.n_vertices,
                    n_edges=topo.n_edges,
                    vertex_coords_nm=geom["vertices_nm"],
                    edge_list=topo.edges,
                    cavity_radius_nm=cavity_r,
                    cavity_volume_A3=cavity_vol,
                    donor_subtypes=donors,
                    donor_positions_nm=positions,
                    donor_count=nd,
                    denticity=nd,
                    is_macrocyclic=True,
                    linker_chemistry=linker,
                    target_metal=target_metal,
                    target_log_ka=log_ka,
                    target_dg_kj=dg,
                    interferent_scores=intf_scores,
                    selectivity_gaps=sel_gaps,
                    min_selectivity_gap=min_gap,
                    worst_interferent=worst,
                    selectivity_grade=_grade(min_gap),
                    design_id=f"PD-{design_id:04d}",
                    pH=pH,
                )
                designs.append(spec)

    # Rank: by selectivity gap (if interferents) or affinity
    if interferents:
        designs.sort(key=lambda d: d.min_selectivity_gap, reverse=True)
    else:
        designs.sort(key=lambda d: d.target_log_ka, reverse=True)

    for i, d in enumerate(designs[:max_results]):
        d.rank = i + 1

    elapsed = time.time() - t0
    return designs[:max_results], elapsed


# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

def _compact_donors(donors):
    from collections import Counter
    c = Counter(donors)
    parts = []
    for st, n in c.most_common():
        short = st.split("_")[1] if "_" in st else st
        el = st.split("_")[0]
        parts.append(f"{n}\u00D7{el}-{short}" if n > 1 else f"{el}-{short}")
    return " + ".join(parts)


def print_pocket_designs(designs, elapsed=0, verbose=False):
    """Pretty-print pocket design results."""
    if not designs:
        print("  No viable designs found.")
        return

    d0 = designs[0]
    print()
    print(f"  MABE Pocket Design Engine")
    print(f"  Target: {d0.target_metal} | pH {d0.pH}")
    if d0.worst_interferent:
        intfs = sorted(set(d.worst_interferent for d in designs if d.worst_interferent))
        all_intfs = set()
        for d in designs:
            all_intfs.update(d.interferent_scores.keys())
        print(f"  Interferents: {', '.join(sorted(all_intfs))}")
    print(f"  Designs evaluated: {len(designs)} | Time: {elapsed:.1f}s")
    print()

    has_sel = any(d.selectivity_gaps for d in designs)

    if has_sel:
        print(f"  {'#':>3}  {'Grd':4}  {'Topology':15}  {'Edge':>5}  {'Cav\u00C5\u00B3':>7}  "
              f"{'Donors':25}  {'logKa':>6}  {'Gap':>6}  {'vs':>5}")
        print(f"  {'─'*90}")
        for d in designs:
            dstr = _compact_donors(d.donor_subtypes)
            print(f"  {d.rank:3d}  {d.selectivity_grade:^4s}  {d.topology_name:15s}  "
                  f"{d.edge_length_nm:5.1f}  {d.cavity_volume_A3:7.0f}  "
                  f"{dstr:25s}  {d.target_log_ka:+6.1f}  "
                  f"{d.min_selectivity_gap:+6.1f}  {d.worst_interferent:>5s}")
    else:
        print(f"  {'#':>3}  {'Topology':15}  {'Edge':>5}  {'Cav\u00C5\u00B3':>7}  "
              f"{'Donors':25}  {'logKa':>7}  {'\u0394G kJ':>7}")
        print(f"  {'─'*80}")
        for d in designs:
            dstr = _compact_donors(d.donor_subtypes)
            print(f"  {d.rank:3d}  {d.topology_name:15s}  "
                  f"{d.edge_length_nm:5.1f}  {d.cavity_volume_A3:7.0f}  "
                  f"{dstr:25s}  {d.target_log_ka:+7.1f}  {d.target_dg_kj:+7.0f}")

    if verbose and designs:
        d = designs[0]
        print()
        print(f"  ── Top Design: {d.design_id} ──")
        print(f"  Topology: {d.topology_name} ({d.n_vertices}V, {d.n_edges}E)")
        print(f"  Edge length: {d.edge_length_nm:.1f} nm")
        print(f"  Cavity: {d.cavity_radius_nm:.2f} nm radius, {d.cavity_volume_A3:.0f} \u00C5\u00B3")
        print(f"  Donors: {d.donor_subtypes}")
        print(f"  Linker: {d.linker_chemistry}")
        print(f"  Target {d.target_metal}: log Ka = {d.target_log_ka:.2f}")
        for intf, gap in sorted(d.selectivity_gaps.items(), key=lambda x: x[1]):
            lk = d.interferent_scores[intf]
            print(f"    vs {intf}: log Ka = {lk:.2f}, gap = {gap:+.2f} ({10**gap:.0f}\u00D7)")
        print(f"  ATHENA input: {d.to_athena_input()['topology']} @ {d.edge_length_nm} nm")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("MABE Pocket Design Engine — Self-Test")
    print("=" * 70)

    # Test 1: Hg2+ selective over Zn2+, Pb2+
    print("\n  Design 1: Hg2+ pocket (selective over Zn2+, Pb2+)")
    designs, t = design_pocket("Hg2+", interferents=["Zn2+", "Pb2+"], pH=5.0,
                                max_results=10)
    print_pocket_designs(designs, t, verbose=True)

    # Test 2: Pb2+ from mine water
    print("\n  Design 2: Pb2+ pocket (selective over Ca2+, Mg2+, Fe3+)")
    designs, t = design_pocket("Pb2+", interferents=["Ca2+", "Mg2+", "Fe3+"],
                                pH=5.0, max_results=10)
    print_pocket_designs(designs, t, verbose=True)

    # Test 3: Cu2+ maximum affinity (no selectivity constraint)
    print("\n  Design 3: Cu2+ maximum affinity")
    designs, t = design_pocket("Cu2+", pH=7.0, max_results=10)
    print_pocket_designs(designs, t)

    # Test 4: ATHENA export
    if designs:
        print("  ATHENA-ready export (top design):")
        import json
        athena = designs[0].to_athena_input()
        print(f"    {json.dumps({k:v for k,v in athena.items() if k != 'vertices'}, indent=2)}")
