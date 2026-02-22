"""
NIST SRD 46 → MABE CalibrationEntry Pipeline

Extracts clean log K values from the NIST stability constants database,
identifies donor atoms using RDKit, maps to MABE's 9 donor subtypes,
and exports in CalibrationEntry format for physics engine calibration.

MABE donor subtypes:
  O_carboxylate, O_hydroxyl, O_phenolate, O_ether, O_phosphoryl
  N_amine, N_pyridine, N_imine
  S_thiolate, S_thioether
  P_phosphine
"""

import sqlite3
import json
import csv
import re
import traceback
from urllib.parse import unquote
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem

# ── NIST Database Path ──────────────────────────────────────────────────────
DB_PATH = "/home/claude/stability-constant-explorer/NIST_SRD_46_ported.db"

# ── Metal Properties Table ──────────────────────────────────────────────────
# From Marcus 1991 / Kepp 2019 / Shannon ionic radii
METAL_PROPERTIES = {
    # formula: (charge, d_electrons, ionic_radius_pm, softness, hsab, hydration_kj)
    "Li(I)":   (1, 0, 76,  0.0, "hard",       -519),
    "Na(I)":   (1, 0, 102, 0.0, "hard",       -365),
    "K(I)":    (1, 0, 138, 0.0, "hard",       -295),
    "Rb(I)":   (1, 0, 152, 0.0, "hard",       -275),
    "Cs(I)":   (1, 0, 167, 0.0, "hard",       -250),
    "Mg(II)":  (2, 0, 72,  0.0, "hard",       -1830),
    "Ca(II)":  (2, 0, 100, 0.0, "hard",       -1505),
    "Sr(II)":  (2, 0, 118, 0.0, "hard",       -1380),
    "Ba(II)":  (2, 0, 135, 0.0, "hard",       -1250),
    "Mn(II)":  (2, 5, 83,  0.3, "borderline", -1760),
    "Fe(II)":  (2, 6, 78,  0.4, "borderline", -1840),
    "Co(II)":  (2, 7, 75,  0.5, "borderline", -1915),
    "Ni(II)":  (2, 8, 69,  0.5, "borderline", -1980),
    "Cu(II)":  (2, 9, 73,  0.6, "borderline", -2010),
    "Zn(II)":  (2, 10, 75, 0.5, "borderline", -1955),
    "Cd(II)":  (2, 10, 95, 0.7, "soft",       -1755),
    "Hg(II)":  (2, 10, 102,0.9, "soft",       -1760),
    "Pb(II)":  (2, 0, 119, 0.6, "borderline", -1425),
    "Ag(I)":   (1, 10,115, 0.8, "soft",       -430),
    "Cu(I)":   (1, 10, 77, 0.7, "soft",       -580),
    "Tl(I)":   (1, 0, 150, 0.5, "soft",       -310),
    "Al(III)": (3, 0, 53,  0.0, "hard",       -4525),
    "Fe(III)": (3, 5, 65,  0.3, "hard",       -4265),
    "Cr(III)": (3, 3, 62,  0.2, "hard",       -4010),
    "Ga(III)": (3, 10, 62, 0.3, "hard",       -4515),
    "In(III)": (3, 10, 80, 0.4, "hard",       -3980),
    "La(III)": (3, 0, 103, 0.0, "hard",       -3145),
    "Ce(III)": (3, 1, 101, 0.0, "hard",       -3200),
    "Nd(III)": (3, 3, 98,  0.0, "hard",       -3280),
    "Eu(III)": (3, 6, 95,  0.0, "hard",       -3360),
    "Gd(III)": (3, 7, 94,  0.0, "hard",       -3425),
    "Dy(III)": (3, 9, 91,  0.0, "hard",       -3520),
    "Yb(III)": (3, 13, 87, 0.0, "hard",       -3570),
    "Lu(III)": (3, 14, 86, 0.0, "hard",       -3600),
    "Th(IV)":  (4, 0, 94,  0.0, "hard",       -5815),
    "U(IV)":   (4, 2, 89,  0.1, "hard",       -6200),
    "UO2(II)": (2, 0, 73,  0.2, "hard",       -1630),  # uranyl
    "VO(II)":  (2, 1, 63,  0.3, "hard",       -1900),  # vanadyl
    "Bi(III)": (3, 0, 103, 0.5, "borderline", -3480),
    "Sc(III)": (3, 0, 75,  0.0, "hard",       -3795),
    "Ti(III)": (3, 1, 67,  0.1, "hard",       -4015),
    "V(III)":  (3, 2, 64,  0.2, "hard",       -4220),
    "Sn(II)":  (2, 0, 93,  0.5, "borderline", -1490),
    "Au(III)": (3, 8, 85,  0.9, "soft",       -4420),
    "Au(I)":   (1, 10, 137,0.95,"soft",       -615),
    "Pd(II)":  (2, 8, 86,  0.8, "soft",       -1910),
    "Pt(II)":  (2, 8, 80,  0.85,"soft",       -1960),
    "Rh(III)": (3, 6, 67,  0.5, "borderline", -4100),
    "Ir(III)": (3, 6, 68,  0.5, "borderline", -4150),
    "Ru(II)":  (2, 6, 68,  0.5, "borderline", -1920),
    "Ru(III)": (3, 5, 68,  0.4, "borderline", -4050),
    "Os(II)":  (2, 6, 63,  0.5, "borderline", -1950),
}


# ── Donor Subtype Identification ────────────────────────────────────────────

def classify_nitrogen(atom, mol):
    """Classify a nitrogen atom into MABE donor subtypes."""
    idx = atom.GetIdx()
    degree = atom.GetDegree()
    aromatic = atom.GetIsAromatic()
    h_count = atom.GetTotalNumHs()
    neighbors = [mol.GetAtomWithIdx(n.GetIdx()) for n in atom.GetNeighbors()]

    # N in aromatic ring — pyridine-type
    if aromatic:
        # Check if it's in a 6-membered ring (pyridine) vs 5-membered (imidazole)
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if idx in ring:
                if len(ring) == 6:
                    return "N_pyridine"
                elif len(ring) == 5:
                    # Imidazole-type — could be N_amine or N_pyridine depending on position
                    # If H on N, it's the pyrrole-type N (poor donor); if no H, pyridine-type
                    if h_count == 0:
                        return "N_pyridine"  # sp2 lone pair, good donor
                    else:
                        return "N_amine"  # pyrrole N, weaker donor
        return "N_pyridine"  # default aromatic N

    # Check for imine (C=N)
    for bond in atom.GetBonds():
        if bond.GetBondTypeAsDouble() == 2.0:
            other = bond.GetOtherAtom(atom)
            if other.GetSymbol() == "C":
                return "N_imine"

    # Check for nitrile (C≡N)
    for bond in atom.GetBonds():
        if bond.GetBondTypeAsDouble() == 3.0:
            return "N_imine"  # nitrile N acts as sp donor

    # Default: amine (sp3 nitrogen with lone pair)
    return "N_amine"


def classify_oxygen(atom, mol):
    """Classify an oxygen atom into MABE donor subtypes."""
    idx = atom.GetIdx()
    degree = atom.GetDegree()
    h_count = atom.GetTotalNumHs()
    charge = atom.GetFormalCharge()
    aromatic = atom.GetIsAromatic()
    neighbors = [mol.GetAtomWithIdx(n.GetIdx()) for n in atom.GetNeighbors()]

    # Carboxylate: O attached to C that also has another O (C(=O)O or C(=O)[O-])
    if degree == 1:  # terminal O
        for neighbor in neighbors:
            if neighbor.GetSymbol() == "C":
                # Check if C has another O neighbor
                c_neighbors = [mol.GetAtomWithIdx(n.GetIdx()) for n in neighbor.GetNeighbors()]
                o_count = sum(1 for cn in c_neighbors if cn.GetSymbol() == "O" and cn.GetIdx() != idx)
                if o_count >= 1:
                    # This is a carboxylate/carboxylic acid O
                    return "O_carboxylate"
                # Check for P=O (phosphoryl)
            elif neighbor.GetSymbol() == "P":
                return "O_phosphoryl"
            elif neighbor.GetSymbol() == "S":
                return "O_sulfonate"  # maps to O_carboxylate analog
        # Lone terminal O on C with no other O — could be aldehyde/ketone
        for neighbor in neighbors:
            if neighbor.GetSymbol() == "C":
                for bond in atom.GetBonds():
                    if bond.GetBondTypeAsDouble() == 2.0:
                        return "O_carbonyl"

    # Phenolate / phenol: O on aromatic C
    if degree == 1 and charge == -1:
        for neighbor in neighbors:
            if neighbor.GetIsAromatic():
                return "O_phenolate"
    if degree == 1 and h_count >= 1:
        for neighbor in neighbors:
            if neighbor.GetIsAromatic():
                return "O_hydroxyl"  # phenol OH

    # Hydroxyl: O-H not on aromatic C
    if h_count >= 1 and degree <= 2:
        return "O_hydroxyl"

    # Ether: O with two bonds to C, no H
    if degree == 2 and h_count == 0:
        n_c = sum(1 for n in neighbors if n.GetSymbol() == "C")
        if n_c == 2:
            # Check if in ring — crown ether vs open-chain
            return "O_ether"

    # Charged carboxylate [O-]
    if charge == -1:
        for neighbor in neighbors:
            if neighbor.GetSymbol() == "C":
                c_neighbors = [mol.GetAtomWithIdx(n.GetIdx()) for n in neighbor.GetNeighbors()]
                o_count = sum(1 for cn in c_neighbors if cn.GetSymbol() == "O")
                if o_count >= 2:
                    return "O_carboxylate"
        return "O_hydroxyl"  # generic oxyanion

    return "O_ether"  # fallback


def classify_sulfur(atom, mol):
    """Classify a sulfur atom into MABE donor subtypes."""
    h_count = atom.GetTotalNumHs()
    charge = atom.GetFormalCharge()
    degree = atom.GetDegree()

    # Thiolate: S- or S-H
    if charge == -1 or h_count >= 1:
        return "S_thiolate"

    # Thioether: S bonded to two carbons
    if degree == 2 and h_count == 0:
        return "S_thioether"

    # Default
    return "S_thiolate"


def classify_phosphorus(atom, mol):
    """Classify phosphorus."""
    # Phosphine: P bonded to C/H only
    neighbors = [mol.GetAtomWithIdx(n.GetIdx()) for n in atom.GetNeighbors()]
    n_o = sum(1 for n in neighbors if n.GetSymbol() == "O")
    if n_o == 0:
        return "P_phosphine"
    return "O_phosphoryl"  # phosphate — O is the real donor


DONOR_ELEMENTS = {"N", "O", "S", "P"}

# SMARTS patterns for common donor groups (backup identification)
DONOR_SMARTS = {
    "O_carboxylate": "[OX1,OX2H0]C(=O)",  # COO-
    "O_hydroxyl": "[OH]",
    "O_phenolate": "[OX1,OH]c",
    "O_ether": "[OD2]([CX4])[CX4]",
    "N_amine": "[NX3;H2,H1,H0;!$(NC=O);!$([nR])]",
    "N_pyridine": "[nR1]1[cR1][cR1][cR1][cR1][cR1]1",
    "N_imine": "[NX2]=[CX3]",
    "S_thiolate": "[SX1,SH]",
    "S_thioether": "[SX2]([CX4])[CX4]",
}


def identify_donor_atoms(mol):
    """
    Identify all potential donor atoms in a molecule and classify by MABE subtype.
    Returns list of donor subtypes and count per subtype.
    """
    if mol is None:
        return [], {}

    donors = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym == "N":
            donors.append(classify_nitrogen(atom, mol))
        elif sym == "O":
            subtype = classify_oxygen(atom, mol)
            # Skip non-donor oxygens (double-bonded O in carboxylate already counted)
            if subtype:
                donors.append(subtype)
        elif sym == "S":
            donors.append(classify_sulfur(atom, mol))
        elif sym == "P":
            # Phosphorus itself is rarely a metal donor except in phosphines
            subtype = classify_phosphorus(atom, mol)
            if subtype == "P_phosphine":
                donors.append(subtype)

    # Deduplicate consecutive carboxylate O's (both O in COO- are counted, but
    # for coordination, a carboxylate typically provides 1-2 donor atoms)
    # Keep all for now — the physics engine handles denticity

    subtype_counts = Counter(donors)
    return donors, dict(subtype_counts)


def mol_from_nist_mol_string(mol_str):
    """Parse zlib-compressed base64-encoded MOL string from NIST database into RDKit mol."""
    if not mol_str or mol_str.strip() == "":
        return None
    try:
        import zlib, base64
        decoded_url = unquote(mol_str)
        raw = base64.b64decode(decoded_url)
        mol_block = zlib.decompress(raw).decode('utf-8')
        mol = Chem.MolFromMolBlock(mol_block, sanitize=True)
        if mol is None:
            mol = Chem.MolFromMolBlock(mol_block, sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                except:
                    pass  # use unsanitized
        return mol
    except Exception:
        return None


def mol_from_formula(formula):
    """
    Attempt to identify donor atoms from molecular formula alone.
    Returns rough donor list based on element counts.
    This is the fallback when no MOL string is available.
    """
    donors = []
    # Count N, O, S, P from formula
    n_count = 0
    o_count = 0
    s_count = 0
    p_count = 0

    # Parse formula like C10H12N2O8 or C2H8N2
    for match in re.finditer(r'([A-Z][a-z]?)(\d*)', formula):
        elem, count = match.groups()
        count = int(count) if count else 1
        if elem == "N":
            n_count = count
        elif elem == "O":
            o_count = count
        elif elem == "S":
            s_count = count
        elif elem == "P":
            p_count = count

    # Rough heuristic: assign generic donor types
    donors.extend(["N_amine"] * n_count)
    donors.extend(["O_carboxylate"] * o_count)  # conservative — most NIST ligands are carboxylate-heavy
    donors.extend(["S_thiolate"] * s_count)
    if p_count > 0:
        donors.extend(["O_phosphoryl"] * p_count)

    return donors, {"N_amine": n_count, "O_carboxylate": o_count,
                     "S_thiolate": s_count, "O_phosphoryl": p_count}


# ── CalibrationEntry dataclass ──────────────────────────────────────────────

@dataclass
class CalibrationEntry:
    """MABE BackSolve CalibrationEntry — one experimental binding measurement."""
    entry_id: str
    log_K_exp: float
    log_K_type: str  # "K1", "beta2", "beta3"

    # Conditions
    temperature_C: float = 25.0
    ionic_strength_M: float = 0.1
    pH: float = 7.0
    solvent: str = "water"

    # Metal
    metal_formula: str = ""
    metal_charge: int = 0
    metal_d_electrons: int = 0
    metal_softness: float = 0.0
    metal_hsab: str = ""
    metal_ionic_radius_pm: float = 0.0
    metal_hydration_kj: float = 0.0

    # Ligand
    ligand_name: str = ""
    ligand_formula: str = ""
    ligand_class: str = ""
    donor_atoms: list = field(default_factory=list)
    donor_subtypes: list = field(default_factory=list)
    donor_subtype_counts: dict = field(default_factory=dict)
    denticity: int = 0
    chelate_rings: int = 0
    is_macrocyclic: bool = False
    n_anionic_donors: int = 0

    # RDKit identification
    donor_id_method: str = ""  # "rdkit_mol", "formula_heuristic", "ligand_class"

    # Calibration metadata
    phase: str = "phase_1_metal"
    series_id: str = ""
    holdout: bool = False
    confidence: str = "high"
    source: str = "NIST_SRD46"
    nist_ligand_nr: int = 0
    nist_metal_nr: int = 0


# ── Ligand Class → Donor Heuristics ────────────────────────────────────────
# When both MOL string and formula fail, use the NIST ligand class

LIGAND_CLASS_DONORS = {
    "Amino acids": {"N_amine": 1, "O_carboxylate": 2},
    "Amino acids (di-, tri-peptide)": {"N_amine": 2, "O_carboxylate": 2},
    "Amines, diamines, triamines": {"N_amine": 2},
    "Amines (cyclic)": {"N_amine": 2},
    "Amines (polyamine)": {"N_amine": 4},
    "Amides": {"N_amine": 1, "O_carbonyl": 1},
    "Pyridines": {"N_pyridine": 1},
    "Pyridines (polysubstituted)": {"N_pyridine": 2},
    "Bipyridyls": {"N_pyridine": 2},
    "Terpyridines": {"N_pyridine": 3},
    "Phenanthrolines": {"N_pyridine": 2},
    "EDTA and derivatives": {"N_amine": 2, "O_carboxylate": 8},
    "NTA and derivatives": {"N_amine": 1, "O_carboxylate": 6},
    "Diacids": {"O_carboxylate": 4},
    "Dicarboxylic acids": {"O_carboxylate": 4},
    "Carboxylic acids": {"O_carboxylate": 2},
    "Hydroxy acids": {"O_carboxylate": 2, "O_hydroxyl": 1},
    "Catechols": {"O_phenolate": 2},
    "Phenols": {"O_phenolate": 1},
    "Hydroxamic acids": {"O_hydroxyl": 1, "O_carboxylate": 1, "N_amine": 1},
    "Macrocyclic (N-donor)": {"N_amine": 4},
    "Macrocyclic (O-donor)": {"O_ether": 6},
    "Macrocyclic (mixed N/O)": {"N_amine": 2, "O_ether": 2},
    "Crown ethers": {"O_ether": 6},
    "Cryptands": {"N_amine": 2, "O_ether": 6},
    "Porphyrins": {"N_pyridine": 4},
    "Phosphonic acids": {"O_phosphoryl": 2},
    "Phosphates": {"O_phosphoryl": 2},
    "Thioethers": {"S_thioether": 1},
    "Thiols": {"S_thiolate": 1},
    "Dithiocarbamates": {"S_thiolate": 2},
    "Cyanides": {"N_imine": 1},  # CN- is C donor, but treated as N_imine-like
    "Halides": {},  # Cl-, Br-, I- — handled separately as monodentate
    "Oxalate": {"O_carboxylate": 4},
    "Imidazoles": {"N_pyridine": 1},
    "Schiff bases": {"N_imine": 1, "O_phenolate": 1},
    "Thiosemicarbazones": {"N_imine": 1, "S_thiolate": 1, "N_amine": 1},
}

MACROCYCLIC_CLASSES = {
    "Macrocyclic (N-donor)", "Macrocyclic (O-donor)", "Macrocyclic (mixed N/O)",
    "Crown ethers", "Cryptands", "Porphyrins",
}


# ── Main Extraction ────────────────────────────────────────────────────────

def parse_html_metal(html_name):
    """Convert NIST HTML metal name to MABE format: Cu<sup>2+</sup> → Cu(II)"""
    # Strip HTML tags, extract element and charge
    clean = re.sub(r'<[^>]+>', '', html_name).strip()
    # Match patterns like Cu2+, Fe3+, Ag+, UO2+, UO22+
    m = re.match(r'^([A-Za-z0-9]+?)(\d*)([+-])$', clean)
    if not m:
        return clean  # return as-is if can't parse
    elem, charge_str, sign = m.groups()
    charge = int(charge_str) if charge_str else 1
    roman = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}
    return f"{elem}({roman.get(charge, str(charge))})"


def extract_nist_calibration_set(db_path=DB_PATH, max_entries=None):
    """
    Extract high-confidence calibration entries from NIST SRD 46.

    Filters:
    - K-type constants only (log K, not H or S)
    - T = 25°C, I = 0.1 M (standard conditions)
    - K1, beta2, beta3 formation constants only
    - Clean numeric values (no parenthetical estimates)
    - Metals in METAL_PROPERTIES table
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Step 1: K type = constanttypID 3
    k_type_nr = 3
    print(f"Constant type K = {k_type_nr}")

    # Step 2: Target beta definitions
    # 812 = [ML]/[M][L], 840 = [ML2]/[M][L]^2, 872 = [ML3]/[M][L]^3
    target_betas = {812: "K1", 840: "beta2", 872: "beta3"}
    print(f"Target beta definitions: {target_betas}")

    # Step 3: Get metal info — parse HTML names
    cur = conn.execute("SELECT metalID, name_metal FROM metal")
    metal_map = {}  # metalID → (html_name, mabe_name)
    for row in cur:
        html = row["name_metal"]
        mabe = parse_html_metal(html)
        metal_map[row["metalID"]] = (html, mabe)
    print(f"Metals: {len(metal_map)}")

    # Step 4: Get ligand info
    cur = conn.execute("""
        SELECT l.ligandenID, l.name_ligand, l.formula, l.ligand_classNr
        FROM liganden l
    """)
    ligand_info = {}
    for row in cur:
        ligand_info[row["ligandenID"]] = {
            "name": row["name_ligand"] or "",
            "formula": row["formula"] or "",
            "class_nr": row["ligand_classNr"],
        }

    # Step 4b: Get ligand class names
    cur = conn.execute("SELECT ligand_classID, name_ligandclass FROM ligand_class")
    class_map = {row["ligand_classID"]: row["name_ligandclass"] for row in cur}

    # Apply class names to ligand_info
    for lnr, linfo in ligand_info.items():
        linfo["ligclass"] = class_map.get(linfo["class_nr"], "")

    print(f"Ligands with info: {len(ligand_info)}")

    # Step 4c: Get MOL strings from mol_data table
    cur = conn.execute("SELECT ligandenNR, mol_string_encoded FROM mol_data")
    mol_strings = {}
    for row in cur:
        mol_strings[row["ligandenNR"]] = row["mol_string_encoded"] or ""
    print(f"Ligands with MOL strings: {len(mol_strings)}")

    # Step 5: Extract qualifying rows
    beta_nrs = ",".join(str(k) for k in target_betas.keys())
    query = f"""
        SELECT v.ligandenNr, v.metalNr, v.beta_definitionNr,
               v.temperature, v.ionicstrength, v.constant
        FROM verkn_ligand_metal v
        WHERE v.constanttypNr = ?
          AND v.beta_definitionNr IN ({beta_nrs})
          AND v.temperature = '25'
          AND v.ionicstrength = '0.1'
          AND v.constant NOT LIKE '%(%'
          AND v.constant NOT LIKE '%~%'
          AND v.constant != ''
          AND v.constant != '\\N'
    """
    cur = conn.execute(query, (k_type_nr,))
    raw_rows = cur.fetchall()
    print(f"\nRaw qualifying rows: {len(raw_rows)}")

    # Step 6: Parse and build CalibrationEntries
    entries = []
    rdkit_success = 0
    formula_fallback = 0
    class_fallback = 0
    no_donors = 0
    metal_miss = 0

    # Pre-parse MOL strings for all ligands (cache)
    print("\nParsing MOL strings with RDKit...")
    mol_cache = {}
    parsed = 0
    failed = 0
    for lnr, mol_str in mol_strings.items():
        if mol_str.strip():
            mol = mol_from_nist_mol_string(mol_str)
            if mol is not None:
                mol_cache[lnr] = mol
                parsed += 1
            else:
                failed += 1
    print(f"  Parsed: {parsed}, Failed: {failed}")

    for row in raw_rows:
        metal_entry = metal_map.get(row["metalNr"])
        if metal_entry is None:
            metal_miss += 1
            continue
        html_name, metal_name = metal_entry

        # Match metal to properties table
        props = METAL_PROPERTIES.get(metal_name)
        if props is None:
            metal_miss += 1
            continue

        charge, d_elec, radius, softness, hsab, hyd_kj = props

        # Parse log K value
        try:
            log_k = float(row["constant"])
        except (ValueError, TypeError):
            continue

        lnr = row["ligandenNr"]
        linfo = ligand_info.get(lnr, {})
        ligand_name = linfo.get("name", "")
        ligand_formula = linfo.get("formula", "")
        ligand_class = linfo.get("ligclass", "")
        log_k_type = target_betas[row["beta_definitionNr"]]

        # Identify donor atoms — three-tier approach
        donor_subtypes = []
        donor_counts = {}
        method = ""

        # Tier 1: RDKit from MOL string
        if lnr in mol_cache:
            donor_subtypes, donor_counts = identify_donor_atoms(mol_cache[lnr])
            if donor_subtypes:
                method = "rdkit_mol"
                rdkit_success += 1

        # Tier 2: Formula heuristic
        if not donor_subtypes and ligand_formula:
            donor_subtypes, donor_counts = mol_from_formula(ligand_formula)
            if donor_subtypes:
                method = "formula_heuristic"
                formula_fallback += 1

        # Tier 3: Ligand class heuristic
        if not donor_subtypes and ligand_class:
            for cls_key, cls_donors in LIGAND_CLASS_DONORS.items():
                if cls_key.lower() in ligand_class.lower():
                    donor_counts = dict(cls_donors)
                    donor_subtypes = []
                    for dt, cnt in cls_donors.items():
                        donor_subtypes.extend([dt] * cnt)
                    method = "ligand_class"
                    class_fallback += 1
                    break

        if not donor_subtypes:
            no_donors += 1
            method = "none"

        # Count anionic donors
        n_anionic = sum(donor_counts.get(k, 0) for k in
                        ["O_carboxylate", "O_phenolate", "S_thiolate", "O_phosphoryl"])

        # Estimate denticity from donor count
        denticity = len(donor_subtypes) if len(donor_subtypes) <= 8 else 8
        # For K1, denticity = number of donor atoms that can coordinate simultaneously
        # For beta2/beta3, this is the per-ligand denticity

        # Chelate rings ~ denticity - 1 for typical chelators
        chelate_rings = max(0, denticity - 1) if denticity > 1 else 0

        # Macrocyclic check — match NIST ligand class names
        is_macro = False
        macro_keywords = ["macrocycl", "crown", "cryptand", "porphyrin", "cyclam",
                          "cyclen", "aza macrocycl", "oxa macrocycl", "thia macrocycl",
                          "aza-macrocycl", "oxa-macrocycl", "aza-oxa", "aza-thia"]
        lc_lower = ligand_class.lower()
        ln_lower = ligand_name.lower()
        for kw in macro_keywords:
            if kw in lc_lower or kw in ln_lower:
                is_macro = True
                break
        # Also detect via RDKit ring analysis
        if not is_macro and lnr in mol_cache:
            ri = mol_cache[lnr].GetRingInfo()
            for ring in ri.AtomRings():
                if len(ring) >= 9:  # macrocycles have large rings
                    is_macro = True
                    break

        entry = CalibrationEntry(
            entry_id=f"NIST_{metal_name}_{lnr}_{log_k_type}",
            log_K_exp=log_k,
            log_K_type=log_k_type,
            temperature_C=25.0,
            ionic_strength_M=0.1,
            metal_formula=metal_name,
            metal_charge=charge,
            metal_d_electrons=d_elec,
            metal_softness=softness,
            metal_hsab=hsab,
            metal_ionic_radius_pm=radius,
            metal_hydration_kj=hyd_kj,
            ligand_name=ligand_name,
            ligand_formula=ligand_formula,
            ligand_class=ligand_class,
            donor_atoms=[d.split("_")[0] for d in donor_subtypes],
            donor_subtypes=donor_subtypes,
            donor_subtype_counts=donor_counts,
            denticity=denticity,
            chelate_rings=chelate_rings,
            is_macrocyclic=is_macro,
            n_anionic_donors=n_anionic,
            donor_id_method=method,
            series_id=f"{metal_name}_{ligand_class}",
            nist_ligand_nr=lnr,
            nist_metal_nr=row["metalNr"],
        )
        entries.append(entry)

        if max_entries and len(entries) >= max_entries:
            break

    conn.close()

    # Report
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total CalibrationEntries: {len(entries)}")
    print(f"Donor ID methods:")
    print(f"  RDKit MOL:          {rdkit_success}")
    print(f"  Formula heuristic:  {formula_fallback}")
    print(f"  Ligand class:       {class_fallback}")
    print(f"  No donors found:    {no_donors}")
    print(f"  Metal not in table: {metal_miss}")

    # Stats
    metals_seen = Counter(e.metal_formula for e in entries)
    print(f"\nMetals: {len(metals_seen)} unique")
    print(f"Top 10: {metals_seen.most_common(10)}")

    subtypes_seen = Counter()
    for e in entries:
        for st in e.donor_subtypes:
            subtypes_seen[st] += 1
    print(f"\nDonor subtype distribution:")
    for st, cnt in subtypes_seen.most_common():
        print(f"  {st:20s}: {cnt:6d}")

    log_k_range = [e.log_K_exp for e in entries]
    print(f"\nlog K range: [{min(log_k_range):.2f}, {max(log_k_range):.2f}]")
    print(f"log K mean: {sum(log_k_range)/len(log_k_range):.2f}")

    return entries


def deduplicate_entries(entries):
    """
    For duplicate M-L pairs (same metal, same ligand, same K type),
    take the median log K value.
    """
    groups = defaultdict(list)
    for e in entries:
        key = (e.metal_formula, e.nist_ligand_nr, e.log_K_type)
        groups[key].append(e)

    deduped = []
    for key, group in groups.items():
        if len(group) == 1:
            deduped.append(group[0])
        else:
            # Take median
            vals = sorted(g.log_K_exp for g in group)
            median_val = vals[len(vals) // 2]
            best = min(group, key=lambda g: abs(g.log_K_exp - median_val))
            best.log_K_exp = median_val
            best.confidence = "high" if len(group) >= 2 else "medium"
            deduped.append(best)

    print(f"\nDeduplication: {len(entries)} → {len(deduped)} unique M-L-type entries")
    return deduped


def export_json(entries, path):
    """Export to JSON for MABE consumption."""
    data = []
    for e in entries:
        d = asdict(e)
        data.append(d)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Exported {len(data)} entries to {path}")


def export_csv(entries, path):
    """Export summary CSV for quick inspection."""
    fieldnames = [
        "entry_id", "metal_formula", "metal_charge", "metal_hsab",
        "ligand_name", "ligand_class", "log_K_exp", "log_K_type",
        "donor_subtypes", "denticity", "chelate_rings", "is_macrocyclic",
        "n_anionic_donors", "donor_id_method", "confidence"
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for e in entries:
            row = asdict(e)
            row["donor_subtypes"] = "|".join(e.donor_subtypes)
            writer.writerow(row)
    print(f"Exported CSV to {path}")


# ── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("NIST SRD 46 → MABE CalibrationEntry Pipeline")
    print("=" * 60)

    # Extract
    entries = extract_nist_calibration_set()

    # Deduplicate
    entries = deduplicate_entries(entries)

    # Export
    export_json(entries, "/home/claude/nist_calibration_entries.json")
    export_csv(entries, "/home/claude/nist_calibration_summary.csv")

    # Quick validation: check known complexes
    print("\n" + "=" * 60)
    print("VALIDATION — Known Complexes")
    print("=" * 60)
    benchmarks = {
        "Cu(II)_EDTA": ("Cu(II)", 6277, 18.78),
        "Fe(III)_EDTA": ("Fe(III)", 6277, 25.1),
        "Ni(II)_EDTA": ("Ni(II)", 6277, 18.4),
        "Zn(II)_EDTA": ("Zn(II)", 6277, 16.5),
        "Ca(II)_EDTA": ("Ca(II)", 6277, 10.7),
        "Cu(II)_Glycine": ("Cu(II)", 5760, 8.22),
        "Ni(II)_Glycine": ("Ni(II)", 5760, 5.77),
    }

    for name, (metal, lig_nr, expected) in benchmarks.items():
        matches = [e for e in entries if e.metal_formula == metal
                   and e.nist_ligand_nr == lig_nr
                   and e.log_K_type == "K1"]
        if matches:
            val = matches[0].log_K_exp
            delta = val - expected
            status = "OK" if abs(delta) < 1.0 else "CHECK"
            print(f"  {name:20s}: {val:6.2f} (expected {expected:6.2f}, delta={delta:+.2f}) [{status}]")
            print(f"    donors: {matches[0].donor_subtypes}")
        else:
            print(f"  {name:20s}: NOT FOUND")