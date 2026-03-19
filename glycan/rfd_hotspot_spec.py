"""
glycan/rfd_hotspot_spec.py -- RFdiffusion-compatible hotspot residue spec.

Translates a glycan scorer pharmacophore (HB contacts, CH-pi, desolvation
keys) into a protein design constraint file for RFdiffusion / ProteinMPNN.

The output is a HotspotSpec containing:
  - Residue type + interaction type + relative coordinate for each contact
  - Contig string for RFdiffusion backbone generation
  - Validation against known PDB structures (do predicted residues match real ones?)

This does NOT require RFdiffusion at runtime. It produces a spec that the
RFdiffusion pipeline can consume.

Usage:
    from glycan.rfd_hotspot_spec import generate_hotspot_spec
    spec = generate_hotspot_spec("ConA", "Man")
    print(spec.contig_string)
    for hs in spec.hotspots:
        print(hs)

Coordinate geometry from PDB crystal structures.
Residue mapping from MABE glycan contact maps + structural knowledge.

Reference PDB structures:
  ConA:    5CNA (Naismith 1994)
  WGA:     2UVO (Muraki 2002)
  PNA:     2PEL (Banerjee 1996)
  Gal3:    3GAL / 1KJL (Seetharaman 1998)
  Siglec2: 5VKM (Ereno-Orbea 2017)
"""

from dataclasses import dataclass, field
from typing import Optional
import math

from glycan.click_site_predictor import POSITION_CONTACTS


# ── Sugar ring coordinate systems ───────────────────────────────────────
# Idealized 4C1 pyranose ring coordinates (Angstrom).
# Origin at ring centroid. X along C1-O5, Y toward C3, Z up (alpha face).
# These are RELATIVE coordinates for placing protein residues around the sugar.
# Source: CSD average pyranose geometry (Cremer-Pople, φ=0, θ=0 for 4C1).

# OH group positions relative to ring centroid (approx, in Angstrom)
_PYRANOSE_OH_COORDS = {
    "C1": (2.3, 0.0, 0.5),       # anomeric, axial alpha
    "C2_eq": (1.4, 1.3, -0.2),   # equatorial (Glc, Gal at C2)
    "C2_ax": (1.4, 1.3, 0.8),    # axial (Man at C2)
    "C3_eq": (-0.3, 2.1, -0.2),  # equatorial
    "C3_ax": (-0.3, 2.1, 0.8),   # axial
    "C4_eq": (-1.8, 1.3, -0.2),  # equatorial (Glc, Man at C4)
    "C4_ax": (-1.8, 1.3, 0.8),   # axial (Gal at C4)
    "C5": (-1.2, 0.0, 0.0),      # ring carbon, no OH (CH-pi face)
    "C6": (-2.5, -1.0, -0.2),    # primary CH2OH, extended
}

# Alpha face center (for CH-pi aromatic placement)
_ALPHA_FACE_CENTER = (0.0, 0.8, 1.5)  # above ring plane

# Standard interaction distances (Angstrom)
_HB_DISTANCE = 2.8    # O...O or O...N hydrogen bond
_CHP_DISTANCE = 3.8   # ring center to aromatic centroid for CH-pi
_SALT_BRIDGE_DIST = 2.7  # COO-...Arg salt bridge


# ── Desolvation key → preferred residue type mapping ────────────────────
# Which protein residues best provide HB contacts for each sugar OH type.
# Based on PDB survey of lectin binding sites.

_DESOLV_TO_RESIDUE: dict[str, list[str]] = {
    "K_EQ": ["ASP", "ASN", "GLU"],     # equatorial OH: carboxylate/amide acceptors
    "K_AX": ["ARG", "HIS", "ASN"],     # axial OH (e.g., Gal C4): guanidinium/imidazole
    "K_C6": ["ASN", "ARG", "GLU"],     # primary CH2OH: amide or bidentate
    "K_NAC": ["ASN", "GLN"],           # NHAc carbonyl: amide NH donor
    "K_COO": ["ARG", "LYS"],           # carboxylate: salt bridge to guanidinium/amine
}

# CH-pi residue preference by aromatic type
_CHP_RESIDUES = {
    "Trp": "TRP",
    "Tyr": "TYR",
    "Phe": "PHE",
    "anthracene": "TRP",  # closest protein analog of anthracene platform
    "none": None,
}

# Known PDB residue assignments for validation
# Format: (scaffold, ligand, position) -> (residue_3letter, PDB_residue_number, PDB_id)
_KNOWN_PDB_RESIDUES: dict[tuple, tuple] = {
    ("ConA", "Man", "C3"): ("ASP", 208, "5CNA"),
    ("ConA", "Man", "C4"): ("ARG", 228, "5CNA"),
    ("ConA", "Man", "C6"): ("ASN", 14, "5CNA"),
    ("ConA", "Man", "C5_chp"): ("TYR", 12, "5CNA"),
    ("ConA", "Glc", "C3"): ("ASP", 208, "5CNA"),
    ("ConA", "Glc", "C4"): ("ASP", 208, "5CNA"),  # bidentate
    ("ConA", "Glc", "C6"): ("ASN", 14, "5CNA"),
    ("ConA", "Glc", "C5_chp"): ("TYR", 12, "5CNA"),
    ("PNA", "Gal", "C3"): ("ASP", 83, "2PEL"),
    ("PNA", "Gal", "C4"): ("GLY", 108, "2PEL"),  # backbone; Gly not ideal for RFd
    ("PNA", "Gal", "C6"): ("HIS", 121, "2PEL"),
    ("PNA", "Gal", "C5_chp"): ("TRP", 132, "2PEL"),
    ("Gal3", "Gal", "C3"): ("ASN", 160, "1KJL"),
    ("Gal3", "Gal", "C4"): ("ARG", 144, "1KJL"),
    ("Gal3", "Gal", "C5_chp"): ("TRP", 181, "1KJL"),
    ("Gal3", "Gal", "C6"): ("ARG", 144, "1KJL"),  # bidentate
    ("WGA", "GlcNAc", "C5_chp"): ("TYR", 73, "2UVO"),
    ("Siglec2", "Neu5Ac", "C1"): ("ARG", 120, "5VKM"),
}


# ── Hotspot dataclass ───────────────────────────────────────────────────

@dataclass
class HotspotResidue:
    """A single hotspot residue for RFdiffusion."""
    position_label: str           # sugar position (C3, C4, C6, C5_chp, etc.)
    interaction_type: str         # "HB_donor", "HB_acceptor", "CH_pi", "salt_bridge"
    residue_type_3: str           # 3-letter amino acid code
    residue_type_1: str           # 1-letter amino acid code
    coord_x: float                # relative x (Angstrom, sugar-centered frame)
    coord_y: float
    coord_z: float
    distance_to_sugar_A: float    # distance from residue to sugar contact point
    desolv_key: Optional[str]     # MABE desolvation key that motivated this
    pdb_validation: Optional[str] # "MATCH: ASP208 in 5CNA" or None
    confidence: str               # "high" (PDB-validated), "medium" (type match), "design"
    note: str

    def __str__(self):
        val = f" [{self.pdb_validation}]" if self.pdb_validation else ""
        return (f"{self.position_label}: {self.residue_type_3} @ "
                f"({self.coord_x:.1f}, {self.coord_y:.1f}, {self.coord_z:.1f}) "
                f"{self.interaction_type} d={self.distance_to_sugar_A:.1f}A "
                f"[{self.confidence}]{val}")


_AA_1LETTER = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


@dataclass
class HotspotSpec:
    """Complete RFdiffusion-compatible hotspot specification."""
    scaffold: str
    ligand: str
    hotspots: list[HotspotResidue]
    n_hb_contacts: int
    n_chp_contacts: int
    n_validated: int              # hotspots with PDB validation
    pocket_depth_A: float         # approximate pocket depth
    recommended_protein_length: int  # residues for RFd contig

    notes: list[str] = field(default_factory=list)

    @property
    def contig_string(self) -> str:
        """RFdiffusion contig format.

        Format: [Lmin-Lmax/Afixed/Lmin-Lmax/Bfixed/...]
        For glycan binder: one contiguous designed chain with hotspot constraints.
        """
        n = self.recommended_protein_length
        # Fixed residues from hotspots (numbered sequentially)
        fixed = []
        for i, hs in enumerate(self.hotspots):
            fixed.append(f"{hs.residue_type_1}{i+1}")
        fixed_str = ",".join(fixed)
        return f"[{n}-{n}]  # hotspots: {fixed_str}"

    @property
    def hotspot_pdb_lines(self) -> list[str]:
        """Generate pseudo-PDB ATOM lines for hotspot residues.

        These can be used as RFdiffusion hotspot input or PyMOL visualization.
        """
        lines = []
        for i, hs in enumerate(self.hotspots):
            atom_num = i + 1
            res_num = i + 1
            # CA atom at the specified coordinate
            line = (
                f"ATOM  {atom_num:>5d}  CA  {hs.residue_type_3:>3s} A{res_num:>4d}    "
                f"{hs.coord_x:>8.3f}{hs.coord_y:>8.3f}{hs.coord_z:>8.3f}"
                f"  1.00  0.00           C"
            )
            lines.append(line)
        return lines

    @property
    def summary(self) -> str:
        lines = [
            f"RFdiffusion Hotspot Spec: {self.scaffold}/{self.ligand}",
            f"  Hotspots: {len(self.hotspots)} ({self.n_hb_contacts} HB + {self.n_chp_contacts} CH-pi)",
            f"  PDB-validated: {self.n_validated}/{len(self.hotspots)}",
            f"  Pocket depth: ~{self.pocket_depth_A:.0f} A",
            f"  Recommended protein: {self.recommended_protein_length} residues",
            f"  Contig: {self.contig_string}",
        ]
        for hs in self.hotspots:
            lines.append(f"    {hs}")
        return "\n".join(lines)


# ── Main generation function ────────────────────────────────────────────

def generate_hotspot_spec(
    scaffold: str,
    ligand: str,
    protein_length: int = 80,
) -> HotspotSpec:
    """Generate RFdiffusion hotspot spec from glycan contact maps.

    Args:
        scaffold: MABE scaffold name (e.g., "ConA", "Gal3")
        ligand: sugar name (e.g., "Man", "Gal")
        protein_length: target protein length in residues (default 80)

    Returns:
        HotspotSpec with residue positions and RFdiffusion input format.
    """
    key = (scaffold, ligand)
    if key not in POSITION_CONTACTS:
        raise ValueError(f"No position contacts for ({scaffold}, {ligand}). "
                         f"Available: {sorted(POSITION_CONTACTS.keys())}")

    pos_map = POSITION_CONTACTS[key]
    hotspots = []
    n_hb = 0
    n_chp = 0

    for pos, info in pos_map.items():
        hb_count = info["hb"]
        desolv = info["desolv"]
        chp_count = info["chp"]
        note = info["note"]

        # HB contacts -> protein residue hotspots
        if hb_count > 0 and desolv is not None:
            residue_options = _DESOLV_TO_RESIDUE.get(desolv, ["ASN"])
            preferred_res = residue_options[0]  # top preference

            # Get sugar OH coordinates
            coord = _get_oh_coord(pos, scaffold, ligand)

            # Place protein residue at HB distance along the contact vector
            # Vector: from sugar OH outward (away from ring center)
            rx, ry, rz = _place_residue_hb(coord)

            # Check PDB validation
            pdb_key = (scaffold, ligand, pos)
            validation = None
            confidence = "design"
            if pdb_key in _KNOWN_PDB_RESIDUES:
                known_res, known_num, known_pdb = _KNOWN_PDB_RESIDUES[pdb_key]
                if known_res == preferred_res:
                    validation = f"MATCH: {known_res}{known_num} in {known_pdb}"
                    confidence = "high"
                else:
                    validation = f"TYPE_DIFF: predicted {preferred_res}, actual {known_res}{known_num} in {known_pdb}"
                    confidence = "medium"
                    # Use the actual PDB residue instead
                    preferred_res = known_res

            interaction = "salt_bridge" if desolv == "K_COO" else "HB_acceptor"

            hotspots.append(HotspotResidue(
                position_label=pos,
                interaction_type=interaction,
                residue_type_3=preferred_res,
                residue_type_1=_AA_1LETTER.get(preferred_res, "X"),
                coord_x=round(rx, 2),
                coord_y=round(ry, 2),
                coord_z=round(rz, 2),
                distance_to_sugar_A=_HB_DISTANCE,
                desolv_key=desolv,
                pdb_validation=validation,
                confidence=confidence,
                note=note,
            ))
            n_hb += hb_count

        # CH-pi contacts -> aromatic residue hotspots
        if chp_count > 0:
            # Get aromatic residue type from scaffold-level res_type
            from glycan.contact_maps import SCAFFOLD_CONTACTS
            scaffold_entry = SCAFFOLD_CONTACTS.get(scaffold, {}).get(ligand, {})
            res_type_raw = scaffold_entry.get("res_type", "Tyr")
            chp_res_3 = _CHP_RESIDUES.get(res_type_raw, "TRP")

            if chp_res_3 is not None:
                ax, ay, az = _ALPHA_FACE_CENTER
                # Place aromatic above the alpha face
                rx = ax
                ry = ay
                rz = az + _CHP_DISTANCE

                pdb_key = (scaffold, ligand, f"{pos}_chp")
                validation = None
                confidence = "design"
                if pdb_key in _KNOWN_PDB_RESIDUES:
                    known_res, known_num, known_pdb = _KNOWN_PDB_RESIDUES[pdb_key]
                    if known_res == chp_res_3:
                        validation = f"MATCH: {known_res}{known_num} in {known_pdb}"
                        confidence = "high"
                    else:
                        validation = f"TYPE_DIFF: predicted {chp_res_3}, actual {known_res}{known_num} in {known_pdb}"
                        confidence = "medium"
                        chp_res_3 = known_res

                hotspots.append(HotspotResidue(
                    position_label=f"{pos}_chp",
                    interaction_type="CH_pi",
                    residue_type_3=chp_res_3,
                    residue_type_1=_AA_1LETTER.get(chp_res_3, "X"),
                    coord_x=round(rx, 2),
                    coord_y=round(ry, 2),
                    coord_z=round(rz + (n_chp * 0.5), 2),  # offset for multiple CH-pi
                    distance_to_sugar_A=_CHP_DISTANCE,
                    desolv_key=None,
                    pdb_validation=validation,
                    confidence=confidence,
                    note=f"CH-pi stacking above alpha face; {note}",
                ))
                n_chp += 1

    n_validated = sum(1 for hs in hotspots if hs.pdb_validation and "MATCH" in hs.pdb_validation)

    # Pocket depth estimate from deepest contact
    max_depth = max((abs(hs.coord_z) for hs in hotspots), default=5.0)
    pocket_depth = max_depth + _HB_DISTANCE

    notes = [
        f"Generated from MABE contact map: {scaffold}/{ligand}",
        f"Sugar coordinate frame: origin at ring centroid, Z up (alpha face)",
        f"HB distances: {_HB_DISTANCE} A; CH-pi distances: {_CHP_DISTANCE} A",
    ]
    if n_validated > 0:
        notes.append(f"{n_validated} hotspots validated against PDB crystal structures")

    return HotspotSpec(
        scaffold=scaffold,
        ligand=ligand,
        hotspots=hotspots,
        n_hb_contacts=n_hb,
        n_chp_contacts=n_chp,
        n_validated=n_validated,
        pocket_depth_A=round(pocket_depth, 1),
        recommended_protein_length=protein_length,
        notes=notes,
    )


# ── Geometry helpers ────────────────────────────────────────────────────

def _get_oh_coord(position: str, scaffold: str, ligand: str) -> tuple[float, float, float]:
    """Get idealized OH coordinates for a sugar position.

    Uses axial/equatorial distinction from contact map notes.
    """
    key = (scaffold, ligand)
    pos_info = POSITION_CONTACTS.get(key, {}).get(position, {})
    note = pos_info.get("note", "").lower()

    if position == "C1":
        return _PYRANOSE_OH_COORDS["C1"]
    elif position == "C2":
        if "axial" in note:
            return _PYRANOSE_OH_COORDS["C2_ax"]
        return _PYRANOSE_OH_COORDS["C2_eq"]
    elif position == "C3":
        if "axial" in note:
            return _PYRANOSE_OH_COORDS["C3_ax"]
        return _PYRANOSE_OH_COORDS["C3_eq"]
    elif position == "C4":
        if "axial" in note:
            return _PYRANOSE_OH_COORDS["C4_ax"]
        return _PYRANOSE_OH_COORDS["C4_eq"]
    elif position == "C5":
        return _PYRANOSE_OH_COORDS["C5"]
    elif position == "C6":
        return _PYRANOSE_OH_COORDS["C6"]
    elif position == "C7":
        # Glycerol sidechain (Neu5Ac) - extends from C6
        c6 = _PYRANOSE_OH_COORDS["C6"]
        return (c6[0] - 1.5, c6[1] - 0.5, c6[2])
    elif position == "C8":
        c6 = _PYRANOSE_OH_COORDS["C6"]
        return (c6[0] - 3.0, c6[1] - 1.0, c6[2])
    elif position == "C9":
        c6 = _PYRANOSE_OH_COORDS["C6"]
        return (c6[0] - 4.5, c6[1] - 1.5, c6[2])
    else:
        return (0.0, 0.0, 0.0)


def _place_residue_hb(oh_coord: tuple[float, float, float]) -> tuple[float, float, float]:
    """Place a protein residue at HB distance from a sugar OH.

    Direction: radially outward from ring center (0,0,0).
    """
    x, y, z = oh_coord
    r = math.sqrt(x**2 + y**2 + z**2)
    if r < 0.1:
        return (0.0, 0.0, _HB_DISTANCE)

    # Unit vector from ring center through OH
    ux, uy, uz = x/r, y/r, z/r

    # Place residue at HB distance along this vector
    rx = x + ux * _HB_DISTANCE
    ry = y + uy * _HB_DISTANCE
    rz = z + uz * _HB_DISTANCE
    return (rx, ry, rz)


# ── Convenience ─────────────────────────────────────────────────────────

def list_available_pharmacophores() -> list[tuple[str, str]]:
    """Return all (scaffold, ligand) pairs with position contacts."""
    return sorted(POSITION_CONTACTS.keys())
