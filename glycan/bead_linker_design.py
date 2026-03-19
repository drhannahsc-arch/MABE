"""
glycan/bead_linker_design.py -- Linker + bead geometry for magnetic pulldown.

Given a CANDIDATE click site on a sugar, computes:
  1. Minimum linker length to clear receptor surface for a bead of diameter D
  2. Entropic penalty of tethered flexible PEG linker
  3. Recommended PEG length with feasibility verdict
  4. Multivalent enhancement estimate for cell-surface pulldown

Physics basis:
  - Pocket depths from PDB crystal structures (5CNA, 2UVO, 2PEL, 3GAL, Eades 2026)
  - PEG monomer length 3.5 A (polymer crystallography)
  - PEG Kuhn length 7.0 A (Oesterhelt 1999, single-molecule AFM)
  - Linker entropy from Flory chain statistics
  - Multivalent framework from Mammen & Whitesides 1998

De novo pipeline usage:
    design = recommend_peg_length("ConA", "Man", "C1", bead_diameter_nm=50)
    # Returns PEG_n, L_contour, DDG_entropy, feasibility
"""

from dataclasses import dataclass
from typing import Optional
import math

from glycan.click_site_predictor import (
    POSITION_CONTACTS, classify_position, CANDIDATE, ESSENTIAL, RING,
)


# ── Physical constants ──────────────────────────────────────────────────

PEG_MONOMER_LENGTH_A = 3.5    # Angstrom per ethylene glycol unit (crystallographic)
PEG_KUHN_LENGTH_A = 7.0       # Angstrom (Oesterhelt 1999, AFM single-molecule)
SURFACE_CLEARANCE_A = 10.0    # van der Waals clearance margin
RT_298 = 2.479                # kJ/mol at 298 K
CLICK_LINKER_LENGTH_A = 12.0  # CuAAC triazole + short spacer (~1.2 nm)


# ── Pocket depth and exit vectors ───────────────────────────────────────
# Pocket depth: distance from sugar OH attachment point to outermost
# receptor surface atom along the exit vector (Angstrom).
# Source: PDB structures measured from ligand to solvent boundary.
#
# Exit class:
#   axial_out     -- linker exits along pocket normal (best clearance)
#   equatorial_out -- linker exits perpendicular to pocket axis
#   recessed      -- linker exits into a groove (needs extra length)

EXIT_VECTORS = {
    ("ConA", "Man", "C1"):     {"depth_A": 8.0,  "exit": "axial_out",
                                 "note": "Anomeric OMe points straight to solvent in 5CNA"},
    ("ConA", "Man", "C2"):     {"depth_A": 8.0,  "exit": "equatorial_out",
                                 "note": "C2-OH axial in Man; exits past Tyr12 face; shallow angle",
                                 "warning": "C2 is CANDIDATE but Schwarz 1996 shows 2-deoxy-Glc abolishes ConA binding. Geometry gating may make C2 non-functional."},
    ("ConA", "Glc", "C1"):     {"depth_A": 8.0,  "exit": "axial_out",
                                 "note": "Anomeric; clean solvent exit"},
    ("ConA", "Glc", "C2"):     {"depth_A": 8.0,  "exit": "equatorial_out",
                                 "note": "C2-OH equatorial in Glc; exits sideways",
                                 "warning": "C2 is CANDIDATE but Schwarz 1996 shows 2-deoxy-Glc abolishes ConA binding. Geometry gating may make C2 non-functional."},
    ("Davis", "Glc", "C1"):    {"depth_A": 10.0, "exit": "axial_out",
                                 "note": "C1-OH beta exits through cavity mouth; deepest pocket"},
    ("PNA", "Gal", "C1"):      {"depth_A": 7.0,  "exit": "axial_out",
                                 "note": "Anomeric; solvent-facing in 2PEL"},
    ("PNA", "Gal", "C2"):      {"depth_A": 7.0,  "exit": "equatorial_out",
                                 "note": "C2-OH equatorial; exits sideways from Trp132 face"},
    ("Gal3", "Gal", "C1"):     {"depth_A": 6.0,  "exit": "axial_out",
                                 "note": "Anomeric; shallow groove in 3GAL"},
    ("Gal3", "Gal", "C2"):     {"depth_A": 6.0,  "exit": "equatorial_out",
                                 "note": "C2-OH equatorial; exits along groove rim"},
    ("WGA", "GlcNAc", "C1"):   {"depth_A": 7.0,  "exit": "axial_out",
                                 "note": "Anomeric; solvent-facing in 2UVO"},
    # ── Siglec-2 / Neu5Ac ──────────────────────────────────────────────
    # C8, C9 on glycerol sidechain. Extends into solvent from C6.
    # Pocket depth minimal (glycerol is fully solvent-exposed).
    # C9-azido-Neu5Ac is a standard bioorthogonal tool (Prescher 2004).
    ("Siglec2", "Neu5Ac", "C8"): {"depth_A": 4.0, "exit": "axial_out",
                                    "note": "C8-OH on glycerol sidechain; fully solvent-exposed in 5VKM"},
    ("Siglec2", "Neu5Ac", "C9"): {"depth_A": 3.0, "exit": "axial_out",
                                    "note": "C9-OH terminal glycerol; most solvent-exposed; standard click site (C9-azido-Neu5Ac)"},
}

# Exit angle penalty: equatorial exit requires extra path length to clear
# receptor surface (linker must bend around the rim).
_EXIT_PENALTY = {
    "axial_out": 1.0,        # direct path, no penalty
    "equatorial_out": 1.4,   # ~40% longer path (geometric: 1/cos(45))
    "recessed": 2.0,         # double path length for groove escape
}


# ── Core calculations ───────────────────────────────────────────────────

@dataclass
class LinkerDesign:
    scaffold: str
    ligand: str
    position: str
    classification: str         # from click_site_predictor
    bead_diameter_nm: float
    bead_radius_A: float
    pocket_depth_A: float
    exit_class: str
    L_min_A: float              # minimum linker length (Angstrom)
    L_min_nm: float
    peg_n_min: int              # minimum PEG units for L_contour >= 1.5 * L_min
    peg_n_recommended: int      # recommended PEG units (L_contour >= 2.0 * L_min for safety)
    L_contour_recommended_A: float
    ddG_entropy_kJ: float       # entropic penalty at recommended PEG length
    feasible: bool
    infeasibility_reason: Optional[str]
    warning: Optional[str]
    note: str


def compute_min_linker_length(scaffold: str, ligand: str, position: str,
                               bead_diameter_nm: float) -> dict:
    """Compute minimum linker length to clear receptor surface for a bead.

    Returns dict with L_min_A, exit_class, pocket_depth_A, bead_radius_A.
    """
    key = (scaffold, ligand, position)
    if key not in EXIT_VECTORS:
        raise ValueError(f"No exit vector data for ({scaffold}, {ligand}, {position})")

    ev = EXIT_VECTORS[key]
    pocket_depth = ev["depth_A"]
    exit_class = ev["exit"]
    exit_penalty = _EXIT_PENALTY[exit_class]

    bead_radius_A = bead_diameter_nm * 10.0 / 2.0  # nm -> A, diameter -> radius

    # Minimum linker: must span from OH position (inside pocket) through
    # pocket opening and out to bead surface, plus clearance.
    # The click chemistry handle adds ~12 A (triazole + short spacer).
    L_min_A = (pocket_depth * exit_penalty + SURFACE_CLEARANCE_A +
               bead_radius_A + CLICK_LINKER_LENGTH_A)

    return {
        "L_min_A": round(L_min_A, 1),
        "pocket_depth_A": pocket_depth,
        "exit_class": exit_class,
        "exit_penalty": exit_penalty,
        "bead_radius_A": bead_radius_A,
    }


def compute_linker_entropy(L_contour_A: float, L_min_A: float, T: float = 298.0) -> float:
    """Entropic penalty for constraining a flexible PEG linker.

    From Flory chain statistics: penalty for restricting end-to-end
    distance to >= L_min when contour length is L_contour.

    Returns DDG in kJ/mol (positive = destabilizing).
    """
    if L_contour_A <= 0:
        return float('inf')
    if L_min_A <= 0:
        return 0.0
    if L_min_A >= L_contour_A:
        return float('inf')  # physically impossible: linker too short

    RT = 8.314e-3 * T  # kJ/mol
    ratio = L_min_A / L_contour_A

    # Flory chain: entropy penalty diverges as ratio -> 1
    # Using the inverse Langevin approximation for worm-like chains:
    # DDG ≈ (3/2) RT [ratio^2 + (1/2)(ratio^4) / (1 - ratio^2)]
    # Simplified for ratio < 0.9: DDG ≈ (3/2) RT ratio^2 / (1 - ratio^2)
    if ratio > 0.95:
        return 50.0  # effectively infinite (taut chain)

    ddG = 1.5 * RT * (ratio ** 2) / (1.0 - ratio ** 2)
    return round(ddG, 3)


def peg_contour_length(n_units: int) -> float:
    """Contour length of PEG_n in Angstrom."""
    return n_units * PEG_MONOMER_LENGTH_A


def peg_units_for_length(L_target_A: float) -> int:
    """Minimum PEG units to achieve a contour length >= L_target."""
    return math.ceil(L_target_A / PEG_MONOMER_LENGTH_A)


def recommend_peg_length(scaffold: str, ligand: str, position: str,
                          bead_diameter_nm: float) -> LinkerDesign:
    """Full linker design recommendation for bead pulldown.

    Checks:
    1. Position must be CANDIDATE (from click_site_predictor)
    2. Computes minimum linker length for bead clearance
    3. Recommends PEG length with entropy < 2 kJ/mol
    4. Returns feasibility verdict
    """
    # Check classification
    cls = classify_position(scaffold, ligand, position)
    warning = None

    if cls.classification == RING:
        return LinkerDesign(
            scaffold=scaffold, ligand=ligand, position=position,
            classification=cls.classification,
            bead_diameter_nm=bead_diameter_nm, bead_radius_A=0,
            pocket_depth_A=0, exit_class="none",
            L_min_A=0, L_min_nm=0,
            peg_n_min=0, peg_n_recommended=0,
            L_contour_recommended_A=0, ddG_entropy_kJ=0,
            feasible=False,
            infeasibility_reason="C5 is ring carbon, no OH to replace",
            warning=None, note="")

    if cls.classification == ESSENTIAL:
        return LinkerDesign(
            scaffold=scaffold, ligand=ligand, position=position,
            classification=cls.classification,
            bead_diameter_nm=bead_diameter_nm, bead_radius_A=0,
            pocket_depth_A=0, exit_class="none",
            L_min_A=0, L_min_nm=0,
            peg_n_min=0, peg_n_recommended=0,
            L_contour_recommended_A=0, ddG_entropy_kJ=0,
            feasible=False,
            infeasibility_reason=f"{position} is ESSENTIAL (removal abolishes binding)",
            warning=None, note=cls.note)

    # CANDIDATE position — compute linker geometry
    key = (scaffold, ligand, position)
    if key not in EXIT_VECTORS:
        return LinkerDesign(
            scaffold=scaffold, ligand=ligand, position=position,
            classification=cls.classification,
            bead_diameter_nm=bead_diameter_nm, bead_radius_A=0,
            pocket_depth_A=0, exit_class="unknown",
            L_min_A=0, L_min_nm=0,
            peg_n_min=0, peg_n_recommended=0,
            L_contour_recommended_A=0, ddG_entropy_kJ=0,
            feasible=False,
            infeasibility_reason="No exit vector data for this position",
            warning=None, note=cls.note)

    ev = EXIT_VECTORS[key]
    warning = ev.get("warning")

    geom = compute_min_linker_length(scaffold, ligand, position, bead_diameter_nm)
    L_min = geom["L_min_A"]

    # PEG sizing: contour >= 1.5x L_min for < 2 kJ/mol entropy
    # Recommended: contour >= 2.0x L_min for comfortable margin
    peg_n_min = peg_units_for_length(1.5 * L_min)
    peg_n_recommended = peg_units_for_length(2.0 * L_min)

    L_contour_rec = peg_contour_length(peg_n_recommended)
    ddG_entropy = compute_linker_entropy(L_contour_rec, L_min)

    # Feasibility: linker is commercially available up to PEG ~5000 units
    # (MW ~220 kDa). Beyond that, use dextran or other polymers.
    feasible = True
    infeasibility_reason = None
    if peg_n_recommended > 5000:
        feasible = False
        infeasibility_reason = f"PEG_{peg_n_recommended} exceeds practical synthesis limit (~PEG_5000)"

    return LinkerDesign(
        scaffold=scaffold,
        ligand=ligand,
        position=position,
        classification=cls.classification,
        bead_diameter_nm=bead_diameter_nm,
        bead_radius_A=geom["bead_radius_A"],
        pocket_depth_A=geom["pocket_depth_A"],
        exit_class=geom["exit_class"],
        L_min_A=round(L_min, 1),
        L_min_nm=round(L_min / 10.0, 1),
        peg_n_min=peg_n_min,
        peg_n_recommended=peg_n_recommended,
        L_contour_recommended_A=round(L_contour_rec, 1),
        ddG_entropy_kJ=round(ddG_entropy, 3),
        feasible=feasible,
        infeasibility_reason=infeasibility_reason,
        warning=warning,
        note=ev["note"],
    )


# ── Multivalent enhancement ─────────────────────────────────────────────

@dataclass
class MultivalentEstimate:
    bead_diameter_nm: float
    bead_contact_area_nm2: float
    sugar_spacing_nm: float
    n_sugars_in_contact: int
    receptor_density_per_um2: float
    n_receptors_under_bead: int
    effective_valency: int
    enhancement_log10: float    # log10(K_eff / K_mono)
    note: str


def estimate_multivalent_enhancement(
    bead_diameter_nm: float,
    sugar_spacing_nm: float = 5.0,
    receptor_density_per_um2: float = 1000.0,
) -> MultivalentEstimate:
    """Estimate multivalent enhancement for bead-cell binding.

    Assumes bead sits on cell surface with a contact patch.
    Contact area approximated as a spherical cap where bead-membrane
    gap < linker reach (~50 nm for typical PEG).

    Args:
        bead_diameter_nm: bead diameter in nm
        sugar_spacing_nm: average spacing between sugars on bead surface
        receptor_density_per_um2: receptor copies per um^2 on cell surface
    """
    R_bead = bead_diameter_nm / 2.0  # nm

    # Contact patch: spherical cap where gap < linker_reach
    # For linker_reach << R_bead: contact area ~ pi * R_bead * linker_reach * 2
    linker_reach_nm = 50.0  # typical PEG_150 reach in nm
    if bead_diameter_nm < 2 * linker_reach_nm:
        # Small bead: entire hemisphere is in contact
        contact_area = 2 * math.pi * R_bead ** 2
    else:
        # Large bead: spherical cap approximation
        # h = R - sqrt(R^2 - r^2) where r = sqrt(2*R*linker_reach)
        # Area = 2*pi*R*h ≈ 2*pi*R*linker_reach (for linker_reach << R)
        contact_area = 2 * math.pi * R_bead * linker_reach_nm

    # Sugars in contact area
    sugar_footprint = sugar_spacing_nm ** 2
    n_sugars = max(1, int(contact_area / sugar_footprint))

    # Receptors under bead
    # Contact area in um^2
    contact_area_um2 = contact_area / 1e6
    n_receptors = max(1, int(contact_area_um2 * receptor_density_per_um2))

    # Effective valency: limited by whichever is fewer
    effective_valency = min(n_sugars, n_receptors)

    # Enhancement: Mammen/Whitesides framework
    # For N independent equivalent sites: K_eff ~ K_mono * (c_eff)^(N-1)
    # where c_eff is the effective local concentration of the next sugar
    # near the next receptor. For flexible linkers on a bead:
    # c_eff ~ 1 / (4/3 * pi * linker_reach^3) in molecules/nm^3
    # Converting: ~1 mM for 50 nm reach
    # Each additional contact adds ~ RT * ln(c_eff / K_d)
    # Simplified: enhancement ~ 10^(0.5 * N) for moderate affinity systems
    # (Mammen 1998, empirical for polyvalent systems)
    if effective_valency <= 1:
        enhancement_log10 = 0.0
    else:
        enhancement_log10 = round(0.5 * (effective_valency - 1), 1)
        # Cap at realistic maximum (~10^6 enhancement observed experimentally)
        enhancement_log10 = min(enhancement_log10, 6.0)

    return MultivalentEstimate(
        bead_diameter_nm=bead_diameter_nm,
        bead_contact_area_nm2=round(contact_area, 1),
        sugar_spacing_nm=sugar_spacing_nm,
        n_sugars_in_contact=n_sugars,
        receptor_density_per_um2=receptor_density_per_um2,
        n_receptors_under_bead=n_receptors,
        effective_valency=effective_valency,
        enhancement_log10=enhancement_log10,
        note=f"Contact patch {contact_area:.0f} nm^2; valency limited by {'receptors' if n_receptors < n_sugars else 'sugars'}",
    )


# ── Convenience: full design recommendation ─────────────────────────────

@dataclass
class PulldownDesign:
    linker: LinkerDesign
    multivalent: MultivalentEstimate
    summary: str


def design_pulldown(scaffold: str, ligand: str, position: str,
                     bead_diameter_nm: float,
                     sugar_spacing_nm: float = 5.0,
                     receptor_density_per_um2: float = 1000.0) -> PulldownDesign:
    """Complete pulldown design: linker + multivalency."""
    linker = recommend_peg_length(scaffold, ligand, position, bead_diameter_nm)
    mv = estimate_multivalent_enhancement(bead_diameter_nm, sugar_spacing_nm,
                                           receptor_density_per_um2)

    if not linker.feasible:
        summary = f"NOT FEASIBLE: {linker.infeasibility_reason}"
    else:
        summary = (
            f"{scaffold}/{ligand} @ {position}: "
            f"PEG_{linker.peg_n_recommended} linker "
            f"({linker.L_contour_recommended_A/10:.0f} nm contour), "
            f"DDG_entropy={linker.ddG_entropy_kJ:.2f} kJ/mol, "
            f"exit={linker.exit_class}, "
            f"multivalent enhancement ~10^{mv.enhancement_log10:.1f}"
        )
        if linker.warning:
            summary += f" [WARNING: {linker.warning}]"

    return PulldownDesign(linker=linker, multivalent=mv, summary=summary)


# ── Oligosaccharide detection ───────────────────────────────────────────

# Oligosaccharide ligand names contain linkage notation or multi-unit markers.
# Monosaccharides: Man, Glc, Gal, GlcNAc, GalNAc, Fru, 2dGlc
# Oligosaccharides: "1->2 diMan", "(GlcNAc)2", "triMan", "LacNAc"
_OLIGO_MARKERS = ["->", "(", "tri", "di", "Lac"]

def is_oligosaccharide(ligand: str) -> bool:
    """Check if a ligand name represents an oligosaccharide."""
    for marker in _OLIGO_MARKERS:
        if marker in ligand:
            # Exclude "2dGlc" which contains "d" but is a monosaccharide
            return True
    return False


# ── Reducing-end linker design ──────────────────────────────────────────
# For oligosaccharides, click handle attaches at the reducing-end C1.
# This is universally CANDIDATE (solvent-exposed, no pharmacophore contacts)
# and always axial_out (anomeric position points into solvent).
#
# Reducing-end pocket depth is shallow (~5 A) because the binding site
# grabs the non-reducing end; the reducing end extends into solvent.

_REDUCING_END_DEPTH_A = 5.0  # conservative; reducing end is at solvent boundary


def recommend_peg_reducing_end(
    scaffold: str,
    ligand: str,
    bead_diameter_nm: float,
) -> LinkerDesign:
    """Linker design for oligosaccharide via reducing-end C1 attachment.

    Reducing-end C1 is always:
    - CANDIDATE (solvent-exposed, no contacts to receptor)
    - axial_out (anomeric position)
    - Shallow pocket depth (~5 A)

    This is how neoglycoconjugates are made in practice.
    """
    bead_radius_A = bead_diameter_nm * 10.0 / 2.0
    pocket_depth = _REDUCING_END_DEPTH_A
    exit_penalty = _EXIT_PENALTY["axial_out"]  # 1.0

    L_min_A = pocket_depth * exit_penalty + SURFACE_CLEARANCE_A + bead_radius_A + CLICK_LINKER_LENGTH_A

    peg_n_min = peg_units_for_length(1.5 * L_min_A)
    peg_n_recommended = peg_units_for_length(2.0 * L_min_A)
    L_contour_rec = peg_contour_length(peg_n_recommended)
    ddG_entropy = compute_linker_entropy(L_contour_rec, L_min_A)

    feasible = True
    infeasibility_reason = None
    if peg_n_recommended > 5000:
        feasible = False
        infeasibility_reason = f"PEG_{peg_n_recommended} exceeds practical synthesis limit (~PEG_5000)"

    return LinkerDesign(
        scaffold=scaffold,
        ligand=ligand,
        position="C1_reducing",
        classification=CANDIDATE,
        bead_diameter_nm=bead_diameter_nm,
        bead_radius_A=bead_radius_A,
        pocket_depth_A=pocket_depth,
        exit_class="axial_out",
        L_min_A=round(L_min_A, 1),
        L_min_nm=round(L_min_A / 10.0, 1),
        peg_n_min=peg_n_min,
        peg_n_recommended=peg_n_recommended,
        L_contour_recommended_A=round(L_contour_rec, 1),
        ddG_entropy_kJ=round(ddG_entropy, 3),
        feasible=feasible,
        infeasibility_reason=infeasibility_reason,
        warning=None,
        note=f"Reducing-end C1 of {ligand}; always solvent-exposed (neoglycoconjugate attachment)",
    )


def design_pulldown_reducing_end(
    scaffold: str,
    ligand: str,
    bead_diameter_nm: float,
    sugar_spacing_nm: float = 5.0,
    receptor_density_per_um2: float = 1000.0,
) -> PulldownDesign:
    """Complete pulldown design for oligosaccharide via reducing-end attachment."""
    linker = recommend_peg_reducing_end(scaffold, ligand, bead_diameter_nm)
    mv = estimate_multivalent_enhancement(bead_diameter_nm, sugar_spacing_nm,
                                           receptor_density_per_um2)

    if not linker.feasible:
        summary = f"NOT FEASIBLE: {linker.infeasibility_reason}"
    else:
        summary = (
            f"{scaffold}/{ligand} @ C1_reducing: "
            f"PEG_{linker.peg_n_recommended} linker "
            f"({linker.L_contour_recommended_A/10:.0f} nm contour), "
            f"DDG_entropy={linker.ddG_entropy_kJ:.2f} kJ/mol, "
            f"exit=axial_out (reducing end), "
            f"multivalent enhancement ~10^{mv.enhancement_log10:.1f}"
        )

    return PulldownDesign(linker=linker, multivalent=mv, summary=summary)
