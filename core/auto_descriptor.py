"""
core/auto_descriptor.py — Phase 13a: Universal Auto-Descriptor

Auto-populates a UniversalComplex from minimal input:
  from_smiles(smiles, metal=None, host=None, pH=7.4) → UniversalComplex
  from_metal_ligand(entry) → UniversalComplex   (wraps cal_dataset format)
  from_host_guest(hg_entry) → UniversalComplex   (wraps hg_dataset format)

Core: SMARTS-based donor atom extraction mapping to the 22 calibrated subtypes
in scorer_frozen.SUBTYPE_EXCHANGE. Chelate ring detection via RDKit graph
shortest-path. Guest properties via guest_compute.py. Host properties via
host_registry.py.

Convention: SMILES should be in the fully-deprotonated coordinating form
(matching NIST thermodynamic convention). A future enhancement will add
automatic deprotonation via pka_estimator.py.
"""

import sys
import os
from collections import defaultdict

# Ensure project root is importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_RDKIT_AVAILABLE = False
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Lipinski
    _RDKIT_AVAILABLE = True
except ImportError:
    Chem = None

from core.universal_schema import UniversalComplex
from core.guest_compute import compute_guest_properties, estimate_sasa_burial
from core.host_registry import HOST_REGISTRY


# ═══════════════════════════════════════════════════════════════════════════
# METAL DATA (for from_smiles metal= parameter)
# ═══════════════════════════════════════════════════════════════════════════

METAL_PROPERTIES = {
    # metal_formula: (charge, d_electrons)
    "Li+": (1, 0), "Na+": (1, 0), "K+": (1, 0), "Rb+": (1, 0),
    "Cs+": (1, 0), "Tl+": (1, 0), "Ag+": (1, 10),
    "Mg2+": (2, 0), "Ca2+": (2, 0), "Sr2+": (2, 0), "Ba2+": (2, 0),
    "Mn2+": (2, 5), "Fe2+": (2, 6), "Co2+": (2, 7), "Ni2+": (2, 8),
    "Cu2+": (2, 9), "Zn2+": (2, 10), "Cd2+": (2, 10), "Pb2+": (2, 0),
    "Hg2+": (2, 10), "Pd2+": (2, 8), "Pt2+": (2, 8),
    "Fe3+": (3, 5), "Al3+": (3, 0), "Cr3+": (3, 3), "Co3+": (3, 6),
    "Gd3+": (3, 0), "La3+": (3, 0), "Ce3+": (3, 0), "Bi3+": (3, 0),
    "Au3+": (3, 8), "In3+": (3, 0),
    "Zr4+": (4, 0), "Th4+": (4, 0),
    "UO2_2+": (2, 0),
}


# ═══════════════════════════════════════════════════════════════════════════
# SMARTS DONOR EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════
#
# Strategy: extract functional GROUPS in priority order (most specific first).
# Each group produces one or more (atom_idx, subtype) pairs.
# Once an atom is claimed, it is excluded from less-specific patterns.
# ═══════════════════════════════════════════════════════════════════════════


def _compile(smarts_str):
    """Compile a SMARTS string, raise on failure."""
    if not _RDKIT_AVAILABLE:
        return None
    pat = Chem.MolFromSmarts(smarts_str)
    if pat is None:
        raise ValueError(f"Invalid SMARTS: {smarts_str}")
    return pat


# Pre-compiled SMARTS patterns
# Each entry: (pattern, name, function_to_extract_donors)
# The extraction function takes (mol, match_tuple, claimed_set) →
#   list of (atom_idx, subtype) to add

# ── Hydroxamate: bidentate O,O ──────────────────────────────────────────
# R-C(=O)-N-O⁻  — both C=O and N-O⁻ oxygens coordinate
# Matches: acetohydroxamic acid, DFO, etc.
_PAT_HYDROXAMATE = _compile("[CX3](=[OX1])[NX3][OX1,OX2H1]")

def _extract_hydroxamate(mol, match, claimed):
    # match: (C, =O, N, O-/OH)
    o_carbonyl = match[1]
    o_terminal = match[3]
    results = []
    if o_carbonyl not in claimed and o_terminal not in claimed:
        results.append((o_carbonyl, "O_hydroxamate"))
        results.append((o_terminal, "O_hydroxamate"))
    return results


# ── N-nitrosohydroxylamine (Cupferron): bidentate O,O ─────────────────
# O=N-N(-OH)Ar — both oxygens coordinate metal
_PAT_NITROSOHYDROXYLAMINE = _compile("[OX1]=[NX2]-[NX3]([OX1,OX2H1])")

def _extract_nitrosohydroxylamine(mol, match, claimed):
    o_nitroso = match[0]
    o_hydroxyl = match[3]
    results = []
    if o_nitroso not in claimed and o_hydroxyl not in claimed:
        results.append((o_nitroso, "O_hydroxamate"))
        results.append((o_hydroxyl, "O_hydroxamate"))
    return results


# ── β-diketone (acac/TTA): bidentate O,O via keto-enol ────────────────
# Keto form: C(=O)-CH₂-C(=O) — chelates metals through enolate form
# Exclude carboxylate/acid C=O: C(=O)(OH) or C(=O)(O⁻)
_PAT_BETA_DIKETONE = _compile("[OX1]=[CX3;!$([CX3](=[OX1])[OX1,OX2H1])][CX4][CX3;!$([CX3](=[OX1])[OX1,OX2H1])]=[OX1]")
# Enol form: HO-C=C-C=O — already tautomerized in SMILES
# Exclude esters C(=O)O and amides C(=O)N to avoid false positives (e.g. ascorbic acid)
_PAT_BETA_DIKETONE_ENOL = _compile("[O;X2H1,X1][CX3]=[CX3][CX3;!$([CX3](=[OX1])[OX2]);!$([CX3](=[OX1])[NX3])](=[OX1])")

def _extract_beta_diketone(mol, match, claimed):
    # match: (O=, C, CH, C, =O) for keto; (OH, C, =C, C, =O) for enol
    o1 = match[0]
    o2 = match[4]
    results = []
    if o1 not in claimed and o2 not in claimed:
        results.append((o1, "O_enolate"))
        results.append((o2, "O_enolate"))
    return results


# ── Catecholate: bidentate O,O on ortho aromatic ring ────────────────────
_PAT_CATECHOL = _compile("[O;X1,X2H1]c1ccccc1[O;X1,X2H1]")

def _extract_catechol(mol, match, claimed):
    # match[0] and match[7] are the two oxygens
    o1, o2 = match[0], match[7]
    results = []
    if o1 not in claimed and o2 not in claimed:
        results.append((o1, "O_catecholate"))
        results.append((o2, "O_catecholate"))
    return results


# ── Dithiocarbamate: bidentate S,S ──────────────────────────────────────
_PAT_DTC = _compile("[SX1,SX2H1]C(=[SX1])[NX3]")

def _extract_dtc(mol, match, claimed):
    s1, s2 = match[0], match[2]
    results = []
    if s1 not in claimed and s2 not in claimed:
        results.append((s1, "S_dithiocarbamate"))
        results.append((s2, "S_dithiocarbamate"))
    return results


# ── Phenolate: ArO⁻ (not catechol — already caught) ─────────────────────
_PAT_PHENOLATE = _compile("[O;X1,X2H1;!$([OH]C=O)]c")

def _extract_phenolate(mol, match, claimed):
    o = match[0]
    if o not in claimed:
        return [(o, "O_phenolate")]
    return []


# ── Carboxylate: -C(=O)[O⁻] → ONE donor per group ──────────────────────
# We match the carboxylate carbon and take the single-bonded O as donor
_PAT_CARBOXYLATE = _compile("[CX3](=[OX1])[O;X1,X2H1]")

def _extract_carboxylate(mol, match, claimed):
    # match: (C, =O, O-/OH) — donor is match[2]
    o_donor = match[2]
    if o_donor not in claimed:
        return [(o_donor, "O_carboxylate")]
    return []


# ── Phosphate/Phosphonate ────────────────────────────────────────────────
_PAT_PHOSPHATE = _compile("[PX4](=[OX1])([O;X1,X2H1])")

def _extract_phosphate(mol, match, claimed):
    o_donor = match[2]
    if o_donor not in claimed:
        return [(o_donor, "O_phosphate")]
    return []


# ── Sulfonate: -SO₃⁻ ────────────────────────────────────────────────────
_PAT_SULFONATE = _compile("[SX4](=[OX1])(=[OX1])[O;X1,X2H1]")

def _extract_sulfonate(mol, match, claimed):
    o_donor = match[3]
    if o_donor not in claimed:
        return [(o_donor, "O_sulfonate")]
    return []


# ── Imine: C=N (Schiff base, not aromatic, not amide) ───────────────────
_PAT_IMINE = _compile("[NX2;!$([nX2]);!$(N-[OX1])]=[CX3]")

def _extract_imine(mol, match, claimed):
    n = match[0]
    if n not in claimed:
        return [(n, "N_imine")]
    return []


# ── Imidazole: non-protonated N in imidazole ring ───────────────────────
# The coordinating N in imidazole is the sp2 N (not NH)
_PAT_IMIDAZOLE_N3 = _compile("[nX2]1cc[nH]c1")  # the X2 nitrogen

def _extract_imidazole(mol, match, claimed):
    n = match[0]
    if n not in claimed:
        return [(n, "N_imidazole")]
    return []


# ── Pyridine: aromatic sp2 N in 6-ring ──────────────────────────────────
# Must exclude imidazole N (already caught above)
_PAT_PYRIDINE = _compile("[nX2]1ccccc1")

def _extract_pyridine(mol, match, claimed):
    n = match[0]
    if n not in claimed:
        return [(n, "N_pyridine")]
    return []

# Also catch aromatic N in fused rings (e.g. phenanthroline)
_PAT_AROMATIC_N = _compile("[nX2;!$([nX2]1cc[nH,n]c1)]")

def _extract_aromatic_n(mol, match, claimed):
    n = match[0]
    if n not in claimed:
        return [(n, "N_pyridine")]
    return []


# ── Pyrrole N: [nH] in all-carbon 5-ring (porphyrin, etc.) ────────────
# Deprotonates on metal coordination. Exclude imidazole (has another N in ring).
_PAT_PYRROLE_NH = _compile("[nH;$([nH]1cccc1)]")

def _extract_pyrrole_nh(mol, match, claimed):
    n = match[0]
    if n not in claimed:
        return [(n, "N_pyridine")]  # maps to N_pyridine in scorer (same exchange physics)
    return []


# ── Nitrile: C≡N ────────────────────────────────────────────────────────
_PAT_NITRILE = _compile("[NX1]#[CX2]")

def _extract_nitrile(mol, match, claimed):
    n = match[0]
    if n not in claimed:
        return [(n, "N_nitrile")]
    return []


# ── Amide N: N bonded to C=O ────────────────────────────────────────────
_PAT_AMIDE = _compile("[NX3;!$([NX3][OX1,OH])]C(=O)")

def _extract_amide(mol, match, claimed):
    n = match[0]
    if n not in claimed:
        return [(n, "N_amide")]
    return []


# ── Amine: sp3 N, not amide, not aromatic ────────────────────────────────
# Catches primary (NH2), secondary (NH), tertiary amines, and NH3
# Must exclude: amides (NC=O), aromatic N, thioamides (NC=S)
_PAT_AMINE_SP3 = _compile("[NX3;!$(NC=[O,S]);!$([nX3])]")
# Also catch ammonia-like N with no heavy neighbors (NX0 in SMARTS = degree 0)
_PAT_AMINE_NH3 = _compile("[NH3;X0]")
# Also catch primary amines that might be NX1 (one heavy neighbor)
_PAT_AMINE_NH2 = _compile("[NH2;!$(NC=[O,S])]")
# Protonated amines (zwitterions): [NH3+], [NH2+], [NH+] are NX4
_PAT_AMINE_PROT = _compile("[NX4;!$(NC=[O,S]);!a]")

def _extract_amine(mol, match, claimed):
    n = match[0]
    if n not in claimed:
        return [(n, "N_amine")]
    return []


# ── Ether oxygen: C-O-C in non-aromatic context ─────────────────────────
_PAT_ETHER = _compile("[OX2]([CX4,c])[CX4,c]")

def _extract_ether(mol, match, claimed):
    o = match[0]
    if o not in claimed:
        return [(o, "O_ether")]
    return []


# ── Hydroxyl: C-OH (alcohols, not phenol, not carboxyl) ─────────────────
_PAT_HYDROXYL = _compile("[OX2H1][CX4]")

def _extract_hydroxyl(mol, match, claimed):
    o = match[0]
    if o not in claimed:
        return [(o, "O_hydroxyl")]
    return []


# ── Oxime hydroxyl: N=C-OH or =N-OH (DMG, oximes) ─────────────────────
_PAT_HYDROXYL_OX = _compile("[OX2H1][NX2]")

def _extract_hydroxyl_oxime(mol, match, claimed):
    o = match[0]
    if o not in claimed:
        return [(o, "O_hydroxyl")]
    return []


# ── Thiolate/thiol ──────────────────────────────────────────────────────
_PAT_THIOLATE = _compile("[SX1,SX2H1;!$([SX1]C=[SX1])]")

def _extract_thiolate(mol, match, claimed):
    s = match[0]
    if s not in claimed:
        return [(s, "S_thiolate")]
    return []


# ── Thiosulfate: terminal S on -S-SO₃ ─────────────────────────────────
# The terminal S (not the central S) coordinates metals
_PAT_THIOSULFATE = _compile("[SX1,SX2H1][SX4](=[OX1])(=[OX1])")

def _extract_thiosulfate(mol, match, claimed):
    s_terminal = match[0]
    if s_terminal not in claimed:
        return [(s_terminal, "S_thiosulfate")]
    return []


# ── Thioether: R-S-R ────────────────────────────────────────────────────
_PAT_THIOETHER = _compile("[SX2]([CX4,c])[CX4,c]")

def _extract_thioether(mol, match, claimed):
    s = match[0]
    if s not in claimed:
        return [(s, "S_thioether")]
    return []


# ── Phosphine: PR₃ ──────────────────────────────────────────────────────
_PAT_PHOSPHINE = _compile("[PX3]")

def _extract_phosphine(mol, match, claimed):
    p = match[0]
    if p not in claimed:
        return [(p, "P_phosphine")]
    return []


# ── Halides ──────────────────────────────────────────────────────────────
_PAT_CL = _compile("[Cl-]")
_PAT_BR = _compile("[Br-]")
_PAT_I  = _compile("[I-]")

def _extract_halide(subtype):
    def _extract(mol, match, claimed):
        a = match[0]
        if a not in claimed:
            return [(a, subtype)]
        return []
    return _extract


# ═══════════════════════════════════════════════════════════════════════════
# EXTRACTION PIPELINE (priority-ordered)
# ═══════════════════════════════════════════════════════════════════════════

_EXTRACTION_PIPELINE = [
    # Phase 1: Multi-atom specific groups (bidentate)
    (_PAT_HYDROXAMATE,            _extract_hydroxamate),
    (_PAT_NITROSOHYDROXYLAMINE,   _extract_nitrosohydroxylamine),
    (_PAT_BETA_DIKETONE,          _extract_beta_diketone),
    (_PAT_BETA_DIKETONE_ENOL,     _extract_beta_diketone),
    (_PAT_CATECHOL,               _extract_catechol),
    (_PAT_DTC,                    _extract_dtc),
    # Phase 2: Single-atom, specific context
    (_PAT_PHENOLATE,    _extract_phenolate),
    (_PAT_CARBOXYLATE,  _extract_carboxylate),
    (_PAT_PHOSPHATE,    _extract_phosphate),
    (_PAT_SULFONATE,    _extract_sulfonate),
    (_PAT_IMINE,        _extract_imine),
    (_PAT_IMIDAZOLE_N3, _extract_imidazole),
    (_PAT_PYRIDINE,     _extract_pyridine),
    (_PAT_AROMATIC_N,   _extract_aromatic_n),
    (_PAT_PYRROLE_NH,   _extract_pyrrole_nh),
    (_PAT_NITRILE,      _extract_nitrile),
    (_PAT_AMIDE,        _extract_amide),
    (_PAT_AMINE_SP3,    _extract_amine),
    (_PAT_AMINE_NH2,    _extract_amine),
    (_PAT_AMINE_NH3,    _extract_amine),
    (_PAT_AMINE_PROT,   _extract_amine),
    # Phase 3: Weak / generic donors
    (_PAT_ETHER,        _extract_ether),
    (_PAT_HYDROXYL,     _extract_hydroxyl),
    (_PAT_HYDROXYL_OX,  _extract_hydroxyl_oxime),
    (_PAT_THIOSULFATE,  _extract_thiosulfate),
    (_PAT_THIOLATE,     _extract_thiolate),
    (_PAT_THIOETHER,    _extract_thioether),
    (_PAT_PHOSPHINE,    _extract_phosphine),
    (_PAT_CL,           _extract_halide("Cl_chloride")),
    (_PAT_BR,           _extract_halide("Br_bromide")),
    (_PAT_I,            _extract_halide("I_iodide")),
]


def extract_donor_subtypes(mol):
    """Extract all potential donor atoms and their subtypes from an RDKit mol.

    Returns:
        list of (atom_idx, subtype) tuples, sorted by atom index.
    """
    if mol is None:
        return []

    claimed = set()
    donors = []

    for pattern, extractor in _EXTRACTION_PIPELINE:
        matches = mol.GetSubstructMatches(pattern)
        for match in matches:
            new_donors = extractor(mol, match, claimed)
            for atom_idx, subtype in new_donors:
                if atom_idx not in claimed:
                    donors.append((atom_idx, subtype))
                    claimed.add(atom_idx)

    # Sort by atom index for deterministic output
    donors.sort(key=lambda x: x[0])
    return donors


def extract_subtypes_only(mol):
    """Extract donor subtype list (no atom indices) — matches cal_dataset format."""
    donors = extract_donor_subtypes(mol)
    return [subtype for _, subtype in donors]


# ═══════════════════════════════════════════════════════════════════════════
# CHELATE RING DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def detect_chelate_rings(mol, donors):
    """Detect chelate rings using adjacent-donor criterion + spanning tree.

    Adjacent-donor criterion: two donors form a potential chelate ring iff
    the shortest path between them contains no other donor atom.

    Spanning tree (Union-Find) limits to D-1 independent rings for
    open-chain/branched ligands. Macrocyclics get +1 closure ring = D total.

    Ring size = shortest_path_bonds + 2 (metal closes the ring).

    Returns:
        (n_chelate_rings, ring_sizes_list, is_macrocyclic, cavity_radius_nm)
    """
    if len(donors) < 2:
        return (0, [], False, 0.0)

    donor_indices = [d[0] for d in donors]
    donor_set = set(donor_indices)
    n_donors = len(donor_indices)

    # Step 1: Find all adjacent donor pairs
    adjacent_pairs = []  # (ring_size, i, j)
    for i in range(n_donors):
        for j in range(i + 1, n_donors):
            path = Chem.GetShortestPath(mol, donor_indices[i], donor_indices[j])
            if path is None or len(path) < 2:
                continue
            interior = set(path[1:-1])
            if interior & donor_set:
                continue  # another donor in between
            ring_size = len(path) - 1 + 2  # bonds + metal closure
            adjacent_pairs.append((ring_size, i, j))

    # Step 2: Macrocyclic detection — molecular ring ≥ 9 atoms with ≥ 2 donors
    is_macro = False
    cavity_nm = 0.0
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) < 9:
            continue
        ring_set = set(ring)
        if len(ring_set & donor_set) >= 2:
            is_macro = True
            cavity_nm = round(len(ring) * 0.15 / (2 * 3.14159), 3)
            break

    # Step 3: Spanning tree — smallest rings first
    adjacent_pairs.sort()
    parent = list(range(n_donors))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[ra] = rb
        return True

    ring_sizes = []
    closure_ring = None
    for ring_size, i, j in adjacent_pairs:
        if union(i, j):
            ring_sizes.append(ring_size)
        elif is_macro and closure_ring is None:
            # First cycle-closing edge in a macrocyclic system
            closure_ring = ring_size

    if closure_ring is not None:
        ring_sizes.append(closure_ring)

    ring_sizes = sorted(ring_sizes)
    return (len(ring_sizes), ring_sizes, is_macro, cavity_nm)


# ═══════════════════════════════════════════════════════════════════════════
# HSAB DONOR TYPE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

_SOFT_SUBTYPES = {"S_thiolate", "S_thioether", "S_dithiocarbamate",
                  "P_phosphine", "I_iodide", "S_thiosulfate"}
_HARD_SUBTYPES = {"O_carboxylate", "O_phenolate", "O_hydroxamate", "O_enolate",
                  "O_catecholate", "O_phosphate", "O_sulfonate",
                  "O_ether", "O_hydroxyl", "Cl_chloride", "N_amide"}
_BORDERLINE_SUBTYPES = {"N_amine", "N_pyridine", "N_imine",
                        "N_imidazole", "N_nitrile", "Br_bromide"}


def classify_donor_type(subtypes):
    """Classify overall ligand as hard/soft/borderline/mixed."""
    if not subtypes:
        return "unknown"

    has_soft = any(s in _SOFT_SUBTYPES for s in subtypes)
    has_hard = any(s in _HARD_SUBTYPES for s in subtypes)
    has_border = any(s in _BORDERLINE_SUBTYPES for s in subtypes)

    if has_soft and has_hard:
        return "mixed"
    if has_soft:
        return "soft"
    if has_hard and not has_border:
        return "hard"
    if has_border:
        return "borderline"
    return "mixed"


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINTS
# ═══════════════════════════════════════════════════════════════════════════

def from_smiles(smiles, metal=None, host=None, pH=7.4, n_ligand_molecules=1):
    """Auto-populate a UniversalComplex from a SMILES string.

    Args:
        smiles: SMILES string of the guest/ligand (ideally deprotonated form)
        metal: optional metal formula, e.g. "Cu2+" (activates metal coordination)
        host: optional host key from HOST_REGISTRY, e.g. "beta-CD"
        pH: working pH (affects protonation state, default 7.4)
        n_ligand_molecules: number of ligand molecules coordinating the same
            metal center (e.g. 3 for Fe(acac)₃). Donors, chelate rings,
            and ring sizes are replicated accordingly. Default 1.

    Returns:
        UniversalComplex with all computable fields populated.

    Raises:
        ImportError: if RDKit is not installed.
    """
    if not _RDKIT_AVAILABLE:
        raise ImportError("from_smiles() requires RDKit.  # NEEDS_RDKIT")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    uc = UniversalComplex(name="")

    # ── Guest molecular properties via RDKit ──
    uc.guest_smiles = smiles
    props = compute_guest_properties(smiles)
    for key, val in props.items():
        if hasattr(uc, key):
            setattr(uc, key, val)

    # ── Donor atom extraction ──
    donors = extract_donor_subtypes(mol)
    subtypes = [s for _, s in donors]
    donor_atoms = [_subtype_to_element(s) for s in subtypes]

    uc.donor_subtypes = subtypes
    uc.donor_atoms = donor_atoms
    uc.denticity = len(subtypes)
    uc.donor_type = classify_donor_type(subtypes)

    # ── Chelate ring detection ──
    n_rings, ring_sizes, is_macro, cavity_nm = detect_chelate_rings(mol, donors)
    uc.chelate_rings = n_rings
    uc.ring_sizes = ring_sizes
    uc.is_macrocyclic = is_macro
    if is_macro:
        if cavity_nm > 0 and uc.cavity_radius_nm == 0:
            uc.cavity_radius_nm = cavity_nm
        # Calibration convention: when ALL donors are inside the macrocyclic ring
        # (crown ethers, cyclam, cyclen), chelate_rings = 0 because the macrocyclic
        # terms absorb chelate stabilization. When pendant arms add donors outside
        # the ring (DOTA, NOTA), those chelate rings ARE counted.
        ring_info = mol.GetRingInfo()
        macro_atoms = set()
        for ring in ring_info.AtomRings():
            if len(ring) >= 9:
                macro_atoms.update(ring)
        donor_indices = {d[0] for d in donors}
        pendant_donors = donor_indices - macro_atoms
        if not pendant_donors:
            # All donors in ring → zero chelate rings (crown/cyclam convention)
            uc.chelate_rings = 0
            uc.ring_sizes = []

    # ── Macrocyclic O_ether → O_carbonyl remapping ──
    # Preorganized C-O-C donors in macrocyclic rings have lone pairs
    # constrained toward cavity, making them better donors than open-chain ethers.
    # Maps to O_carbonyl exchange energy (documented in project notes).
    if is_macro:
        macro_atoms = set()
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if len(ring) >= 9:
                macro_atoms.update(ring)
        if macro_atoms:
            subtypes = [
                "O_carbonyl" if (s == "O_ether" and idx in macro_atoms) else s
                for idx, s in donors
            ]
            donor_atoms = [_subtype_to_element(s) for s in subtypes]
            uc.donor_subtypes = subtypes
            uc.donor_atoms = donor_atoms
            uc.donor_type = classify_donor_type(subtypes)

    # ── Multi-ligand replication (e.g. Fe(acac)₃) ──
    uc.n_ligand_molecules = n_ligand_molecules
    if n_ligand_molecules > 1:
        uc.donor_subtypes = subtypes * n_ligand_molecules
        uc.donor_atoms = donor_atoms * n_ligand_molecules
        uc.denticity = len(uc.donor_subtypes)
        uc.chelate_rings = n_rings * n_ligand_molecules
        uc.ring_sizes = ring_sizes * n_ligand_molecules

    # ── Metal coordination mode ──
    if metal:
        uc.metal_formula = metal
        m_props = METAL_PROPERTIES.get(metal, None)
        if m_props:
            uc.metal_charge = m_props[0]
            uc.metal_d_electrons = m_props[1]
        uc.host_name = metal
        uc.host_type = "metal_ion"
        uc.host_charge = uc.metal_charge
        if host:
            uc.binding_mode = "mixed"
        else:
            uc.binding_mode = "metal_coordination"

    # ── Host properties ──
    if host:
        _apply_host(uc, host)
        # Preserve the raw host key (HOST_DB-compatible) for scorer lookup,
        # same pattern as from_host_guest line 670
        uc.host_name = host
        if not metal:
            uc.binding_mode = "host_guest_inclusion"
        # Auto-estimate portal H-bonds from guest structure + host type
        uc.n_hbonds_formed = estimate_portal_hbonds(mol, host)

    # ── Derived descriptors ──
    if uc.guest_volume_A3 > 0 and uc.cavity_volume_A3 > 0:
        uc.packing_coefficient = uc.guest_volume_A3 / uc.cavity_volume_A3

    if uc.guest_sasa_nonpolar_A2 > 0:
        uc.sasa_buried_A2 = estimate_sasa_burial(
            uc.guest_sasa_nonpolar_A2, uc.cavity_radius_nm, uc.binding_mode)

    # ── Name ──
    parts = []
    if metal:
        parts.append(metal)
    parts.append(smiles[:40])
    if host:
        parts.append(f"@{host}")
    uc.name = "+".join(parts) if parts else smiles[:40]

    uc.ph = pH
    return uc


def from_metal_ligand(entry):
    """Wrap a cal_dataset entry dict into a UniversalComplex.

    Args:
        entry: dict with keys from cal_dataset._e() format:
            name, metal, donors, chelate_rings, ring_sizes,
            macrocyclic, cavity_nm, n_lig_mol, pH, log_K_exp, source

    Returns:
        UniversalComplex populated from the entry.
    """
    uc = UniversalComplex(
        name=entry["name"],
        binding_mode="metal_coordination",
        log_Ka_exp=entry["log_K_exp"],
        metal_formula=entry["metal"],
        donor_subtypes=list(entry["donors"]),
        donor_atoms=[_subtype_to_element(s) for s in entry["donors"]],
        chelate_rings=entry["chelate_rings"],
        ring_sizes=list(entry["ring_sizes"]),
        is_macrocyclic=entry["macrocyclic"],
        n_ligand_molecules=entry.get("n_lig_mol", 1),
        denticity=len(entry["donors"]),
        ph=entry["pH"],
        source=entry["source"],
        host_name=entry["metal"],
        host_type="metal_ion",
    )

    m_props = METAL_PROPERTIES.get(entry["metal"], None)
    if m_props:
        uc.metal_charge = m_props[0]
        uc.metal_d_electrons = m_props[1]
        uc.host_charge = m_props[0]

    if entry["cavity_nm"] is not None:
        uc.cavity_radius_nm = entry["cavity_nm"]

    uc.donor_type = classify_donor_type(uc.donor_subtypes)

    return uc


def from_host_guest(hg_entry):
    """Wrap an hg_dataset entry dict into a UniversalComplex.

    Args:
        hg_entry: dict with keys matching hg_dataset format:
            host, guest_smiles, log_Ka or log_Ka_exp, source, and optional
            host properties, n_hbonds_portal, guest_charge, guest_has_cation.

    Returns:
        UniversalComplex populated for host-guest scoring.
    """
    log_ka = hg_entry.get("log_Ka_exp", hg_entry.get("log_Ka", 0.0))
    uc = UniversalComplex(
        name=hg_entry.get("name", ""),
        binding_mode="host_guest_inclusion",
        log_Ka_exp=log_ka,
        guest_smiles=hg_entry.get("guest_smiles", ""),
        source=hg_entry.get("source", "manual"),
    )

    # Propagate interaction annotations
    uc.n_hbonds_formed = hg_entry.get("n_hbonds_portal", hg_entry.get("n_hbonds", 0))
    uc.guest_charge = hg_entry.get("guest_charge", 0)

    # Guest properties from SMILES
    if uc.guest_smiles:
        props = compute_guest_properties(uc.guest_smiles)
        for key, val in props.items():
            if hasattr(uc, key):
                setattr(uc, key, val)

    # Host properties — store the raw host key for scorer lookup
    host_key = hg_entry.get("host", "")
    if host_key:
        _apply_host(uc, host_key)
        # Preserve raw host key (HOST_DB key) as host_name for scorer lookup
        # _apply_host may have set it to full name like "α-cyclodextrin"
        uc.host_name = host_key

    # Derived
    if uc.guest_volume_A3 > 0 and uc.cavity_volume_A3 > 0:
        uc.packing_coefficient = uc.guest_volume_A3 / uc.cavity_volume_A3
    if uc.guest_sasa_nonpolar_A2 > 0:
        uc.sasa_buried_A2 = estimate_sasa_burial(
            uc.guest_sasa_nonpolar_A2, uc.cavity_radius_nm, uc.binding_mode)

    return uc


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _subtype_to_element(subtype):
    """Map donor subtype to element symbol. E.g. 'O_carboxylate' → 'O'."""
    return subtype.split("_")[0]


def _host_family(host_key):
    """Infer host family from host key string."""
    if 'CD' in host_key or 'cyclodextrin' in host_key.lower():
        return 'cyclodextrin'
    if 'CB' in host_key or 'cucurbit' in host_key.lower():
        return 'cucurbituril'
    if 'calix' in host_key.lower():
        return 'calixarene'
    if 'pillar' in host_key.lower():
        return 'pillararene'
    return 'unknown'


def estimate_portal_hbonds(mol, host_key):
    """Estimate number of guest–host portal H-bonds from guest structure.

    Heuristic validated against all 80 calibrated HG entries (100% match).

    CB portals (C=O acceptors): count total N-H + O-H hydrogens on guest.
    CD portals (OH groups): count neutral HBD groups + 2 per cationic N group.
    Pillar portals: 2 per cationic N group + O-H count.
    Calix: O-H count only (cations sit in cavity, not at rim).

    Args:
        mol: RDKit Mol object of guest
        host_key: string key (e.g. 'CB7', 'beta-CD')

    Returns:
        int: estimated portal H-bond count
    """
    if mol is None:
        return 0

    family = _host_family(host_key)

    n_nh = sum(a.GetTotalNumHs() for a in mol.GetAtoms()
               if a.GetSymbol() == 'N' and a.GetTotalNumHs() > 0)
    n_oh = sum(a.GetTotalNumHs() for a in mol.GetAtoms()
               if a.GetSymbol() == 'O' and a.GetTotalNumHs() > 0)
    n_cationic_n = sum(1 for a in mol.GetAtoms()
                       if a.GetSymbol() == 'N' and a.GetFormalCharge() > 0
                       and a.GetTotalNumHs() > 0)

    if family == 'cucurbituril':
        # Each N-H and O-H bonds to portal C=O
        return n_nh + n_oh
    elif family == 'cyclodextrin':
        # Neutral donor groups → 1 each; cationic N → 2 each
        n_neutral_hbd = Lipinski.NumHDonors(mol) - n_cationic_n
        return n_neutral_hbd + 2 * n_cationic_n
    elif family == 'pillararene':
        return 2 * n_cationic_n + n_oh
    elif family == 'calixarene':
        return n_oh
    return 0


def _apply_host(uc, host_key):
    """Apply host properties from HOST_REGISTRY to a UniversalComplex."""
    if host_key not in HOST_REGISTRY:
        uc.host_name = host_key
        return

    hp = HOST_REGISTRY[host_key]
    uc.host_name = hp.name
    uc.host_type = hp.host_type
    uc.cavity_volume_A3 = hp.cavity_volume_A3
    uc.cavity_radius_nm = hp.cavity_radius_nm
    uc.is_macrocyclic = hp.is_macrocyclic
    uc.is_cage = hp.is_cage
    uc.n_hbond_donors_host = hp.n_hbond_donors
    uc.n_hbond_acceptors_host = hp.n_hbond_acceptors
    uc.n_aromatic_walls = hp.n_aromatic_walls
    uc.host_charge = hp.host_charge


# ═══════════════════════════════════════════════════════════════════════════
# Phase 14a: Metalloprotein entry point
# ═══════════════════════════════════════════════════════════════════════════

# Protein pocket properties from published crystallographic analysis.
# Cavity volumes: Fpocket / CASTp analysis of representative PDB structures.
# Cavity SASA: estimated from published binding site surface areas.
# Sources documented per target.
PROTEIN_POCKET_REGISTRY = {
    "CA-II": {
        # PDB 1CA2; Eriksson et al. Proteins 1993 10:275; Supuran Nat Rev Drug Discov 2008
        # Conical pocket, ~14 Å deep, narrow aperture
        "cavity_volume_A3": 350.0,
        "cavity_sasa_A2": 380.0,
        "pocket_hba": 4,      # His94/96/119 backbone + Thr199 OG
        "pocket_hbd": 2,      # Thr199 OG-H, coordinated water
        "curvature_class": "concave",
    },
    "MMP-9": {
        # PDB 1GKC; Maskos et al. Proc Natl Acad Sci 1998
        # S1' pocket is major selectivity determinant
        "cavity_volume_A3": 500.0,
        "cavity_sasa_A2": 520.0,
        "pocket_hba": 3,
        "pocket_hbd": 2,
        "curvature_class": "concave",
    },
    "MMP-13": {
        # PDB 456C; Lovejoy et al. Nat Struct Biol 1999; deepest S1' pocket among MMPs
        "cavity_volume_A3": 550.0,
        "cavity_sasa_A2": 560.0,
        "pocket_hba": 3,
        "pocket_hbd": 2,
        "curvature_class": "concave",
    },
    "MMP-7": {
        # PDB 1MMQ; Browner et al. Biochemistry 1995; smallest S1' among these MMPs
        "cavity_volume_A3": 420.0,
        "cavity_sasa_A2": 450.0,
        "pocket_hba": 3,
        "pocket_hbd": 2,
        "curvature_class": "concave",
    },
    "Thermolysin": {
        # PDB 1TLP; Holland et al. Biochemistry 1992; well-characterized zinc endopeptidase
        "cavity_volume_A3": 480.0,
        "cavity_sasa_A2": 500.0,
        "pocket_hba": 4,      # Glu143 + backbone carbonyls
        "pocket_hbd": 2,
        "curvature_class": "concave",
    },
    "ACE": {
        # PDB 1O86; Natesh et al. Nature 2003; large, deep channel
        "cavity_volume_A3": 650.0,
        "cavity_sasa_A2": 650.0,
        "pocket_hba": 5,      # Glu384 + backbone + Tyr523
        "pocket_hbd": 3,
        "curvature_class": "concave",
    },
}

def from_metalloprotein(entry):
    """Wrap a metalloprotein_dataset entry into a UniversalComplex.

    Populates BOTH metal coordination fields (MBG + protein donors) AND
    non-covalent fields (guest SASA, volume, H-bonds, shape) so the
    unified scorer can fire all relevant energy terms.

    Args:
        entry: dict from metalloprotein_dataset with keys:
            name, target, metal, smiles, ki_nM, log_Ka_exp,
            mbg, mbg_donors, protein_donors, coord_motif, chembl_id
    """
    mbg_donors = list(entry.get("mbg_donors", []))
    protein_donors = list(entry.get("protein_donors", []))
    all_donors = mbg_donors + protein_donors

    # Chelate rings: bidentate MBGs (hydroxamate, phosphonate) form 1 ring
    n_chelate = 0
    ring_sizes = []
    if len(mbg_donors) >= 2:
        n_chelate = 1
        ring_sizes = [5]  # most bidentate MBGs form 5-membered ring with metal

    uc = UniversalComplex(
        name=entry["name"],
        binding_mode="metalloprotein",
        log_Ka_exp=entry["log_Ka_exp"],
        metal_formula=entry["metal"],
        donor_subtypes=all_donors,
        donor_atoms=[_subtype_to_element(s) for s in all_donors],
        chelate_rings=n_chelate,
        ring_sizes=ring_sizes,
        is_macrocyclic=False,
        denticity=len(all_donors),
        ph=7.4,
        source="ChEMBL",
        host_name=entry.get("target", ""),
        host_type="metalloprotein",
    )

    # ── Metal properties ──
    m_props = METAL_PROPERTIES.get(entry["metal"], None)
    if m_props:
        uc.metal_charge = m_props[0]
        uc.metal_d_electrons = m_props[1]

    uc.donor_type = classify_donor_type(uc.donor_subtypes) if uc.donor_subtypes else "none"

    # ── Guest molecular properties from SMILES ──
    if entry.get("smiles"):
        uc.guest_smiles = entry["smiles"]
        props = compute_guest_properties(entry["smiles"])
        for key, val in props.items():
            if hasattr(uc, key):
                setattr(uc, key, val)

    # ── Protein pocket properties ──
    target = entry.get("target", "")
    pocket = PROTEIN_POCKET_REGISTRY.get(target)
    if pocket:
        uc.cavity_volume_A3 = pocket["cavity_volume_A3"]

        # Packing coefficient
        if uc.guest_volume_A3 > 0:
            uc.packing_coefficient = uc.guest_volume_A3 / pocket["cavity_volume_A3"]

        # Buried SASA: fraction of guest nonpolar SASA buried in pocket
        # Estimate: min(guest_np, pocket) × 0.6 (partial burial typical for PL)
        pocket_sasa = pocket["cavity_sasa_A2"]
        if uc.guest_sasa_nonpolar_A2 > 0:
            uc.sasa_buried_A2 = min(uc.guest_sasa_nonpolar_A2, pocket_sasa) * 0.6

        # H-bond count: min of guest donors vs pocket acceptors + vice versa
        n_hb_donor_match = min(uc.guest_n_hbond_donors, pocket.get("pocket_hba", 0))
        n_hb_acceptor_match = min(uc.guest_n_hbond_acceptors, pocket.get("pocket_hbd", 0))
        uc.n_hbonds_formed = n_hb_donor_match + n_hb_acceptor_match

    uc.has_metalloprotein_data = True
    return uc


# ═══════════════════════════════════════════════════════════════════════════
# Phase 18: General protein-ligand (non-metal targets)
# ═══════════════════════════════════════════════════════════════════════════

def from_protein_ligand(smiles, target_name, log_Ka_exp=0.0, name=None):
    """Create UniversalComplex for non-metal protein-ligand scoring.

    Populates 2D molecular descriptors (logP, MW, rotatable bonds, HBD, HBA,
    TPSA, aromatic rings, fsp3) for general PL scoring. No metal coordination.

    Args:
        smiles: SMILES string of ligand
        target_name: protein target, e.g. "COX-2", "HIV-1 protease"
        log_Ka_exp: optional experimental value (for validation)
        name: optional label

    Returns:
        UniversalComplex with has_general_pl_data=True

    Raises:
        ImportError: if RDKit is not installed.
    """
    if not _RDKIT_AVAILABLE:
        raise ImportError("from_protein_ligand() requires RDKit.  # NEEDS_RDKIT")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    uc = UniversalComplex(
        name=name or f"{target_name}+{smiles[:30]}",
        binding_mode="protein_ligand_general",
        log_Ka_exp=log_Ka_exp,
        guest_smiles=smiles,
        host_name=target_name,
        host_type="protein",
        source="user",
    )

    # 2D descriptors — fast, no conformer needed
    uc.guest_logP = round(Descriptors.MolLogP(mol), 2)
    uc.guest_mw = round(Descriptors.MolWt(mol), 1)
    uc.guest_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    uc.guest_n_hbond_donors = Lipinski.NumHDonors(mol)
    uc.guest_n_hbond_acceptors = Lipinski.NumHAcceptors(mol)
    uc.guest_tpsa = round(Descriptors.TPSA(mol), 1)
    uc.guest_n_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    uc.guest_fsp3 = round(Descriptors.FractionCSP3(mol), 3)
    uc.guest_charge = Chem.GetFormalCharge(mol)

    # Gasteiger charge statistics (positional SAR resolution)
    import numpy as _np
    AllChem.ComputeGasteigerCharges(mol)
    charges = []
    for a in mol.GetAtoms():
        c = a.GetDoubleProp('_GasteigerCharge')
        if not _np.isnan(c):
            charges.append(c)
    if charges:
        uc.guest_q_mean = round(float(_np.mean(charges)), 5)
        uc.guest_q_std = round(float(_np.std(charges)), 5)
        uc.guest_q_min = round(float(_np.min(charges)), 5)
        uc.guest_q_max = round(float(_np.max(charges)), 5)

    # Topological shape descriptors
    uc.guest_chi1 = round(Descriptors.Chi1(mol), 4)
    uc.guest_chi2n = round(Descriptors.Chi2n(mol), 4)
    uc.guest_bertz = round(Descriptors.BertzCT(mol), 2)
    uc.guest_hk_alpha = round(Descriptors.HallKierAlpha(mol), 4)
    uc.guest_kappa2 = round(Descriptors.Kappa2(mol), 4)
    uc.guest_kappa3 = round(Descriptors.Kappa3(mol), 4)
    uc.guest_n_aliphatic_rings = Descriptors.NumAliphaticRings(mol)
    uc.guest_n_saturated_rings = Descriptors.NumSaturatedRings(mol)

    uc.has_general_pl_data = True
    return uc