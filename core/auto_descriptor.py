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

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Lipinski

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
    # Phase 1: Multi-atom specific groups
    (_PAT_HYDROXAMATE,  _extract_hydroxamate),
    (_PAT_CATECHOL,     _extract_catechol),
    (_PAT_DTC,          _extract_dtc),
    # Phase 2: Single-atom, specific context
    (_PAT_PHENOLATE,    _extract_phenolate),
    (_PAT_CARBOXYLATE,  _extract_carboxylate),
    (_PAT_PHOSPHATE,    _extract_phosphate),
    (_PAT_SULFONATE,    _extract_sulfonate),
    (_PAT_IMINE,        _extract_imine),
    (_PAT_IMIDAZOLE_N3, _extract_imidazole),
    (_PAT_PYRIDINE,     _extract_pyridine),
    (_PAT_AROMATIC_N,   _extract_aromatic_n),
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
_HARD_SUBTYPES = {"O_carboxylate", "O_phenolate", "O_hydroxamate",
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

def from_smiles(smiles, metal=None, host=None, pH=7.4):
    """Auto-populate a UniversalComplex from a SMILES string.

    Args:
        smiles: SMILES string of the guest/ligand (ideally deprotonated form)
        metal: optional metal formula, e.g. "Cu2+" (activates metal coordination)
        host: optional host key from HOST_REGISTRY, e.g. "beta-CD"
        pH: working pH (affects protonation state, default 7.4)

    Returns:
        UniversalComplex with all computable fields populated.
    """
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
    if is_macro and cavity_nm > 0 and uc.cavity_radius_nm == 0:
        uc.cavity_radius_nm = cavity_nm

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
        if not metal:
            uc.binding_mode = "host_guest_inclusion"

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
            host, guest_smiles, log_Ka_exp, source, and optional
            host properties.

    Returns:
        UniversalComplex populated for host-guest scoring.
    """
    uc = UniversalComplex(
        name=hg_entry.get("name", ""),
        binding_mode="host_guest_inclusion",
        log_Ka_exp=hg_entry.get("log_Ka_exp", 0.0),
        guest_smiles=hg_entry.get("guest_smiles", ""),
        source=hg_entry.get("source", "manual"),
    )

    # Guest properties from SMILES
    if uc.guest_smiles:
        props = compute_guest_properties(uc.guest_smiles)
        for key, val in props.items():
            if hasattr(uc, key):
                setattr(uc, key, val)

    # Host properties
    host_key = hg_entry.get("host", "")
    if host_key:
        _apply_host(uc, host_key)

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