"""
core/tier2_descriptors.py — Auto-Populate Tier 2 Interaction Descriptors

Enriches a UniversalComplex with Tier 2 fields from SMILES + host context.
Called automatically by guest_compute.enrich_complex() after base properties.

What gets populated:

  FROM SMILES ALONE (guest-side):
    T1:  guest_polarizability_A3   — atom-additive from Schwerdtfeger/Miller tables
    T4:  n_halogen_bonds, halogen_bond_type, halogen_bond_nucleophile
         — SMARTS detection of C-X σ-hole donors + nucleophilic acceptors
    T6:  guest_formal_charge, guest_ion_radius_A, has_marcus_hydration_dg
         — RDKit formal charge + Shannon radius lookup
    T10: buried_groups — functional group inventory via SMARTS

  FROM SMILES + HOST CONTEXT:
    T2:  n_cation_pi_contacts, cation_pi_type
         — cationic guest + host aromatic walls
    T3:  n_pi_stack_contacts, pi_stack_type
         — aromatic guest + host aromatic walls
    T5:  n_salt_bridges, salt_bridge_z_product
         — opposite charges on guest vs host
    T7:  max_hbond_chain_length, hbond_chain_type
         — estimated from H-bond donor/acceptor adjacency

  NOT AUTO-POPULATED (requires explicit annotation):
    T8:  anion-π (partial: perfluoro detection)
    T9:  metallophilic (requires metal pair input)

All fields default to 0/empty, so terms self-zero if detection fails.
"""

import math

_RDKIT_AVAILABLE = False
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    _RDKIT_AVAILABLE = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════
# REFERENCE DATA
# ═══════════════════════════════════════════════════════════════════════════

# Atomic polarizabilities (Å³) for additive estimate
# Source: Schwerdtfeger & Nagle, Mol. Phys. 117:1200 (2019)
#         Miller KJ, JACS 112:8533 (1990) — group additivity refinement
_ATOM_ALPHA = {
    1:  0.387,   # H (bonded, effective — Miller 1990)
    6:  1.76,    # C
    7:  1.10,    # N
    8:  0.802,   # O
    9:  0.557,   # F
    15: 3.63,    # P
    16: 2.90,    # S
    17: 2.18,    # Cl
    35: 3.05,    # Br
    53: 5.35,    # I
    34: 3.77,    # Se
    14: 5.38,    # Si
}

# Shannon ionic radii (Å) for common ions in 6-coordinate geometry
# Source: Shannon RD. Acta Cryst. A32:751 (1976)
# Key for Born solvation (T6)
_SHANNON_RADII = {
    # Monovalent
    "Li+": 0.76,  "Na+": 1.02,  "K+": 1.38,   "Rb+": 1.52,
    "Cs+": 1.67,  "Ag+": 1.15,  "Cu+": 0.77,  "Au+": 1.37,
    "Tl+": 1.50,  "NH4+": 1.48,
    # Divalent
    "Mg2+": 0.72, "Ca2+": 1.00, "Sr2+": 1.18, "Ba2+": 1.35,
    "Mn2+": 0.83, "Fe2+": 0.78, "Co2+": 0.745, "Ni2+": 0.69,
    "Cu2+": 0.73, "Zn2+": 0.74, "Cd2+": 0.95, "Pb2+": 1.19,
    "Hg2+": 1.02, "Pd2+": 0.86, "Pt2+": 0.80,
    # Trivalent
    "Fe3+": 0.645, "Al3+": 0.535, "Cr3+": 0.615, "Co3+": 0.545,
    "Au3+": 0.70,  "In3+": 0.80,  "Bi3+": 1.03,
    "La3+": 1.032, "Ce3+": 1.01,  "Gd3+": 0.938,
    # Tetravalent
    "Zr4+": 0.72,  "Th4+": 0.94,
}

# Metals with Marcus hydration ΔG data (existing Term 2 handles them)
# Source: Marcus Y. J. Chem. Soc. Faraday Trans. 1991, 87:2995
_MARCUS_METALS = {
    "Li+", "Na+", "K+", "Rb+", "Cs+", "Ag+",
    "Mg2+", "Ca2+", "Sr2+", "Ba2+",
    "Mn2+", "Fe2+", "Co2+", "Ni2+", "Cu2+", "Zn2+",
    "Cd2+", "Pb2+", "Hg2+",
    "Fe3+", "Al3+", "Cr3+", "Co3+", "Au3+",
    "La3+", "Gd3+",
}


# ═══════════════════════════════════════════════════════════════════════════
# SMARTS PATTERNS (compiled once at import)
# ═══════════════════════════════════════════════════════════════════════════

if _RDKIT_AVAILABLE:
    # T4: Halogen bond donors — C-X where X is Cl/Br/I bonded to sp2 or EWG carbon
    # σ-hole requires electron-withdrawing context. We match any C-X, then
    # filter by checking neighboring atom electronegativity.
    _PAT_C_CL = Chem.MolFromSmarts("[Cl;X1][#6]")
    _PAT_C_BR = Chem.MolFromSmarts("[Br;X1][#6]")
    _PAT_C_I  = Chem.MolFromSmarts("[I;X1][#6]")

    # T4: Nucleophilic acceptors for halogen bonds
    _PAT_LONE_PAIR_N = Chem.MolFromSmarts("[#7;X2,X3;!$([#7]~[#8]=[#8])]")  # N with lone pair
    _PAT_LONE_PAIR_O = Chem.MolFromSmarts("[#8;X1,X2]")  # O with lone pair
    _PAT_LONE_PAIR_S = Chem.MolFromSmarts("[#16;X1,X2]")  # S with lone pair

    # T5: Cationic groups (salt bridge donors)
    _PAT_CATIONIC_N  = Chem.MolFromSmarts("[#7+;!$([#7]~[#8]=[#8])]")   # N+, not nitro
    _PAT_GUANIDINIUM = Chem.MolFromSmarts("[#7]C(=[#7+])[#7]")           # guanidinium

    # T5: Anionic groups (salt bridge acceptors)
    _PAT_CARBOXYLATE_ANION = Chem.MolFromSmarts("[OX1]C([OX1])=O")      # COO-
    _PAT_PHOSPHATE_ANION   = Chem.MolFromSmarts("[OX1]P([OX1])(=O)")    # PO4
    _PAT_SULFONATE_ANION   = Chem.MolFromSmarts("[OX1]S([OX1])(=O)=O")  # SO3-

    # T8: Electron-poor aromatics for anion-π
    _PAT_PERFLUOROARYL = Chem.MolFromSmarts("c1(F)c(F)c(F)c(F)c(F)c1F")  # C6F6 / C6F5X

    # T10: Functional group inventory for group desolvation
    _PAT_GROUPS = {
        "OH_primary_eq":    Chem.MolFromSmarts("[OX2H1][CX4H2]"),   # primary -CH2-OH
        "OH_secondary_eq":  Chem.MolFromSmarts("[OX2H1][CX4H1]"),   # secondary -CH-OH
        "NH2":              Chem.MolFromSmarts("[NX3H2;!$(NC=O)]"),  # primary amine
        "NHAc":             Chem.MolFromSmarts("[NX3H1]C(=O)[CX4]"), # acetamide-type
        "COOH":             Chem.MolFromSmarts("[CX3](=O)[OX2H1]"),  # carboxylic acid
        "COO_minus":        Chem.MolFromSmarts("[CX3](=[OX1])[OX1-]"), # carboxylate anion
        "amide_CO":         Chem.MolFromSmarts("[CX3](=[OX1])[NX3]"), # amide C=O
        "ring_O":           Chem.MolFromSmarts("[OX2;R]"),            # ring ether O
        "SH":               Chem.MolFromSmarts("[SX2H1]"),            # thiol
        "S_minus":          Chem.MolFromSmarts("[SX1-]"),             # thiolate
        "NH3_plus":         Chem.MolFromSmarts("[NX4H3+]"),           # ammonium
        "phenyl":           Chem.MolFromSmarts("c1ccccc1"),           # phenyl ring
    }


# ═══════════════════════════════════════════════════════════════════════════
# T1: POLARIZABILITY (atom-additive)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_polarizability(mol):
    """Atom-additive polarizability from Schwerdtfeger/Miller tables.

    Returns total molecular polarizability in Å³.
    """
    alpha = 0.0
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        alpha += _ATOM_ALPHA.get(z, 1.5)  # default 1.5 for unlisted elements
    # Add implicit H contribution
    n_implicit_h = sum(a.GetTotalNumHs() for a in mol.GetAtoms())
    alpha += n_implicit_h * _ATOM_ALPHA.get(1, 0.387)
    return round(alpha, 2)


# ═══════════════════════════════════════════════════════════════════════════
# T4: HALOGEN BOND DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def _detect_halogen_bonds(mol):
    """Detect potential halogen bond donors (C-X with σ-hole) and acceptors.

    Returns dict with n_halogen_bonds, halogen_bond_type, halogen_bond_nucleophile.
    Only counts intermolecular potential — actual bond depends on host.
    For intramolecular, reports the most favorable donor-acceptor pair.
    """
    result = {"n_xb_donors": 0, "donors": [], "n_xb_acceptors": 0}

    for pat, x_type, x_sym in [
        (_PAT_C_I,  "C-I",  "I"),
        (_PAT_C_BR, "C-Br", "Br"),
        (_PAT_C_CL, "C-Cl", "Cl"),
    ]:
        matches = mol.GetSubstructMatches(pat)
        for match in matches:
            halogen_idx = match[0]
            carbon_idx = match[1]
            carbon = mol.GetAtomWithIdx(carbon_idx)

            # σ-hole strength correlates with electron-withdrawing context.
            # Strong σ-hole: C(sp2)-X, aromatic C-X, C(F)-X
            # Weak σ-hole: C(sp3,H-rich)-X
            is_aromatic = carbon.GetIsAromatic()
            is_sp2 = carbon.GetHybridization().name == "SP2"

            # Check for electron-withdrawing neighbors (F, Cl, NO2, CF3, C=O)
            ewg_neighbors = 0
            for nbr in carbon.GetNeighbors():
                if nbr.GetIdx() == halogen_idx:
                    continue
                if nbr.GetSymbol() in ("F", "Cl"):
                    ewg_neighbors += 1
                if nbr.GetSymbol() == "N" and any(
                    b.GetBondTypeAsDouble() == 2.0
                    for b in nbr.GetBonds()
                ):
                    ewg_neighbors += 1  # nitro-like

            has_sigma_hole = is_aromatic or is_sp2 or ewg_neighbors > 0

            if has_sigma_hole:
                result["n_xb_donors"] += 1
                result["donors"].append({
                    "type": x_type,
                    "atom_idx": halogen_idx,
                    "strength": "strong" if (is_aromatic or ewg_neighbors > 0) else "moderate",
                })

    # Count nucleophilic acceptors
    n_acc = 0
    best_acc = "N"  # default
    for pat, acc_type in [(_PAT_LONE_PAIR_N, "N"), (_PAT_LONE_PAIR_O, "O"), (_PAT_LONE_PAIR_S, "S")]:
        matches = mol.GetSubstructMatches(pat)
        n_acc += len(matches)
        if matches and best_acc == "N":
            best_acc = acc_type  # prefer N > O > S ordering
    result["n_xb_acceptors"] = n_acc
    result["best_acceptor"] = best_acc

    return result


# ═══════════════════════════════════════════════════════════════════════════
# T5: SALT BRIDGE — charge detection
# ═══════════════════════════════════════════════════════════════════════════

def _detect_charged_groups(mol):
    """Detect cationic and anionic functional groups for salt bridge potential.

    Returns dict with n_cationic, n_anionic, max_positive_charge, max_negative_charge.
    """
    n_cat = 0
    n_an = 0
    max_pos = 0
    max_neg = 0

    # Cationic
    for pat in [_PAT_CATIONIC_N, _PAT_GUANIDINIUM]:
        n_cat += len(mol.GetSubstructMatches(pat))

    # Also count any atom with positive formal charge
    for atom in mol.GetAtoms():
        fc = atom.GetFormalCharge()
        if fc > 0:
            max_pos = max(max_pos, fc)
        elif fc < 0:
            max_neg = min(max_neg, fc)

    # Anionic
    for pat in [_PAT_CARBOXYLATE_ANION, _PAT_PHOSPHATE_ANION, _PAT_SULFONATE_ANION]:
        n_an += len(mol.GetSubstructMatches(pat))

    return {
        "n_cationic": max(n_cat, sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() > 0)),
        "n_anionic": max(n_an, sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() < 0)),
        "max_positive": max_pos,
        "max_negative": max_neg,
    }


# ═══════════════════════════════════════════════════════════════════════════
# T7: H-BOND CHAIN LENGTH ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════

def _estimate_hbond_chain_length(mol):
    """Estimate longest potential H-bond relay chain in molecule.

    Walks adjacency of H-bond donors and acceptors (atoms that can both
    donate and accept: OH, NH). Two relay atoms are "linked" if separated
    by 1-4 bonds (covers O-C-C-O motif in sugars, N-C-C-N in diamines).
    """
    # Atoms that can relay H-bonds (both donor AND acceptor)
    relay_atoms = set()
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym == "O" and atom.GetTotalNumHs() > 0:  # OH
            relay_atoms.add(atom.GetIdx())
        elif sym == "N" and atom.GetTotalNumHs() > 0:  # NH
            relay_atoms.add(atom.GetIdx())

    if len(relay_atoms) < 2:
        return len(relay_atoms)  # 0 or 1

    # Build relay adjacency graph: two relay atoms are linked if
    # shortest path between them is ≤ 4 bonds (O-C-C-O = 3 bonds typical)
    relay_list = sorted(relay_atoms)
    adj = {r: set() for r in relay_list}
    for i, r1 in enumerate(relay_list):
        for r2 in relay_list[i+1:]:
            path = Chem.GetShortestPath(mol, r1, r2)
            if path is not None and 2 <= len(path) <= 5:  # 1-4 bonds
                # Exclude paths that go through another relay atom (chain must be sequential)
                interior = set(path[1:-1])
                if not (interior & relay_atoms):
                    adj[r1].add(r2)
                    adj[r2].add(r1)

    # BFS to find longest connected chain
    max_chain = 1
    visited_global = set()
    for start in relay_list:
        if start in visited_global:
            continue
        queue = [(start, 1)]
        visited = {start}
        while queue:
            current, depth = queue.pop(0)
            max_chain = max(max_chain, depth)
            for nbr in adj[current]:
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, depth + 1))
        visited_global.update(visited)

    return min(max_chain, 5)  # cap at 5 (cooperativity saturates)


# ═══════════════════════════════════════════════════════════════════════════
# T10: FUNCTIONAL GROUP INVENTORY
# ═══════════════════════════════════════════════════════════════════════════

def _inventory_functional_groups(mol):
    """SMARTS-based inventory of functional groups for desolvation costing.

    Returns list of {"type": str, "burial_fraction": float}.
    burial_fraction starts at 0.0 (unknown until host context).
    """
    groups = []
    claimed = set()  # avoid double-counting atoms

    for group_type, pat in _PAT_GROUPS.items():
        if pat is None:
            continue
        matches = mol.GetSubstructMatches(pat)
        for match in matches:
            # Use first atom in match as the identifier
            key_atom = match[0]
            if key_atom not in claimed:
                groups.append({"type": group_type, "burial_fraction": 0.0})
                claimed.add(key_atom)

    return groups


# ═══════════════════════════════════════════════════════════════════════════
# T8: ANION-π PARTIAL DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def _detect_electron_poor_aromatics(mol):
    """Detect perfluorinated or heavily EWG-substituted aromatics for anion-π."""
    n = len(mol.GetSubstructMatches(_PAT_PERFLUOROARYL))
    return n


# ═══════════════════════════════════════════════════════════════════════════
# MASTER POPULATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def populate_tier2(uc):
    """Auto-populate all Tier 2 descriptor fields on a UniversalComplex.

    Requires uc.guest_smiles to be set. Uses host context from uc fields
    (n_aromatic_walls, host_charge, n_hbond_donors_host, etc.) if available.

    Modifies uc in-place. Safe to call multiple times (idempotent).
    """
    if not _RDKIT_AVAILABLE:
        return uc
    if not uc.guest_smiles:
        return uc

    mol = Chem.MolFromSmiles(uc.guest_smiles)
    if mol is None:
        return uc

    # ── T1: Polarizability ──────────────────────────────────────────────
    if uc.guest_polarizability_A3 == 0.0:
        uc.guest_polarizability_A3 = _compute_polarizability(mol)

    # ── T4: Halogen bonds ───────────────────────────────────────────────
    if uc.n_halogen_bonds == 0:
        xb = _detect_halogen_bonds(mol)
        # Intramolecular: if guest has both donors and acceptors
        if xb["n_xb_donors"] > 0 and xb["n_xb_acceptors"] > 0:
            # Potential intramolecular XB (rare but real)
            pass  # Don't auto-assign — need 3D geometry
        # Intermolecular: guest XB donors interact with host acceptors
        # or guest XB acceptors interact with host XB donors
        if xb["n_xb_donors"] > 0:
            # Guest has σ-hole donors — count how many could contact host
            # Requires host to have nucleophilic sites (N, O lone pairs)
            # Conservative: report donors available, not bonds formed
            uc.n_halogen_bonds = 0  # stays 0 until host context confirms contacts
            # Store detection for downstream use
            if xb["donors"]:
                uc.halogen_bond_type = xb["donors"][0]["type"]
                # If host has acceptors (most hosts do), estimate 1 XB per strong donor
                if (uc.n_hbond_acceptors_host > 0 or
                    uc.binding_mode in ("protein_ligand", "metalloprotein", "protein_ligand_general")):
                    uc.n_halogen_bonds = min(xb["n_xb_donors"],
                                             max(uc.n_hbond_acceptors_host, 1))
                    uc.halogen_bond_nucleophile = "O"  # most common in proteins
                    uc.halogen_bond_angle = 170.0  # assume favorable geometry

    # ── T6: Born solvation ──────────────────────────────────────────────
    if uc.guest_formal_charge == 0:
        uc.guest_formal_charge = Chem.GetFormalCharge(mol)

    # Ion radius — look up if metal, estimate if organic ion
    if uc.guest_ion_radius_A == 0.0:
        if uc.metal_formula and uc.metal_formula in _SHANNON_RADII:
            uc.guest_ion_radius_A = _SHANNON_RADII[uc.metal_formula]
        elif abs(uc.guest_formal_charge) > 0:
            # Organic ion: estimate effective Born radius from molecular volume
            # r_eff ≈ (3V / 4π)^(1/3) where V is van der Waals volume
            vol = getattr(uc, 'guest_volume_A3', 0.0)
            if vol > 0:
                uc.guest_ion_radius_A = round((3.0 * vol / (4.0 * math.pi)) ** (1.0/3.0), 2)

    # Marcus data flag
    if uc.metal_formula:
        uc.has_marcus_hydration_dg = uc.metal_formula in _MARCUS_METALS

    # ── T2: Cation-π ────────────────────────────────────────────────────
    if uc.n_cation_pi_contacts == 0:
        guest_charge = Chem.GetFormalCharge(mol)
        guest_n_cationic = sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() > 0)
        guest_n_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
        host_aromatic_walls = getattr(uc, 'n_aromatic_walls', 0)

        # Case A: Cationic guest + host with aromatic walls (calixarene, cyclophane)
        if guest_charge > 0 and host_aromatic_walls > 0:
            # Each cationic center can interact with aromatic walls
            n_contacts = min(guest_n_cationic, host_aromatic_walls)
            uc.n_cation_pi_contacts = n_contacts
            # Classify type
            if uc.metal_formula:
                uc.cation_pi_type = "alkali_benzene"
            else:
                # Check if NMe3+ / NMe4+ (quaternary ammonium — strongest aqueous cation-π)
                pat_nme = Chem.MolFromSmarts("[NX4;H0;!$(NC=O)]")
                if pat_nme and mol.HasSubstructMatch(pat_nme):
                    uc.cation_pi_type = "organic_cation_benzene"  # NR4+ → generic
                else:
                    uc.cation_pi_type = "ammonium_benzene"
            uc.cation_pi_distance_A = 3.5  # default equilibrium

        # Case B: Metal cation + aromatic walls (metal@calixarene)
        elif uc.metal_formula and host_aromatic_walls > 0:
            uc.n_cation_pi_contacts = min(1, host_aromatic_walls)
            uc.cation_pi_type = "alkali_benzene"
            uc.cation_pi_distance_A = 3.5

    # ── T3: π-π stacking ───────────────────────────────────────────────
    if uc.n_pi_stack_contacts == 0:
        guest_n_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
        host_aromatic_walls = getattr(uc, 'n_aromatic_walls', 0)

        if guest_n_aromatic > 0 and host_aromatic_walls > 0:
            # Each guest aromatic ring can stack with one host aromatic wall
            uc.n_pi_stack_contacts = min(guest_n_aromatic, host_aromatic_walls)
            uc.pi_stack_type = "parallel_displaced"  # most common geometry

    # ── T5: Salt bridge ─────────────────────────────────────────────────
    if uc.n_salt_bridges == 0:
        charges = _detect_charged_groups(mol)
        host_charge = getattr(uc, 'host_charge', 0)

        # Guest cationic + host anionic (e.g., cation + sulfonato-calixarene)
        if charges["n_cationic"] > 0 and host_charge < 0:
            uc.n_salt_bridges = min(charges["n_cationic"], abs(host_charge))
            uc.salt_bridge_z_product = -1 * charges["max_positive"]  # z_A * z_B

        # Guest anionic + host cationic (less common in supramolecular hosts)
        elif charges["n_anionic"] > 0 and host_charge > 0:
            uc.n_salt_bridges = min(charges["n_anionic"], host_charge)
            uc.salt_bridge_z_product = -1 * abs(charges["max_negative"])

        # Intramolecular salt bridges (guest has both + and -)
        # Zwitterions, e.g. amino acids — these form internal salt bridges
        elif charges["n_cationic"] > 0 and charges["n_anionic"] > 0:
            # Internal salt bridges: minimum of positive and negative groups
            uc.n_salt_bridges = min(charges["n_cationic"], charges["n_anionic"])
            uc.salt_bridge_z_product = -1

    # ── T7: H-bond chain length ─────────────────────────────────────────
    if uc.max_hbond_chain_length == 0:
        chain = _estimate_hbond_chain_length(mol)

        # Extend by host H-bond network if host has donors/acceptors
        host_hbd = getattr(uc, 'n_hbond_donors_host', 0)
        host_hba = getattr(uc, 'n_hbond_acceptors_host', 0)
        if host_hbd + host_hba > 0 and chain > 0:
            # Host can extend guest chains by 1-2 links
            chain = min(chain + 1, 5)

        uc.max_hbond_chain_length = chain

        # Classify chain type
        if chain >= 2:
            # Check if amide-dominated
            pat_amide = Chem.MolFromSmarts("[CX3](=[OX1])[NX3]")
            n_amides = len(mol.GetSubstructMatches(pat_amide)) if pat_amide else 0
            # Count ALL hydroxyl groups (primary + secondary + anomeric)
            pat_all_oh = Chem.MolFromSmarts("[OX2H1]")
            n_oh = len(mol.GetSubstructMatches(pat_all_oh)) if pat_all_oh else 0
            if n_amides >= 2:
                uc.hbond_chain_type = "amide"
            elif n_oh >= 3:
                uc.hbond_chain_type = "hydroxyl"
            else:
                uc.hbond_chain_type = "default"

    # ── T8: Anion-π (partial) ───────────────────────────────────────────
    if uc.n_anion_pi_contacts == 0:
        n_e_poor = _detect_electron_poor_aromatics(mol)
        if n_e_poor > 0:
            # Guest has electron-poor aromatic — needs anionic partner
            guest_charge = Chem.GetFormalCharge(mol)
            host_charge = getattr(uc, 'host_charge', 0)
            if host_charge < 0 or guest_charge < 0:
                # Anion available (host is anionic, or guest self-associates)
                uc.n_anion_pi_contacts = n_e_poor
                uc.anion_pi_type = "Cl_perfluoroarene"

    # ── T10: Group desolvation inventory ────────────────────────────────
    if not uc.buried_groups:
        groups = _inventory_functional_groups(mol)
        uc.buried_groups = groups

    # Update burial fractions from host context (runs even on re-call)
    if uc.buried_groups and uc.cavity_volume_A3 > 0:
        pc = getattr(uc, 'packing_coefficient', 0.0)
        if pc > 0.3:
            burial_est = min(0.8, pc)  # tighter fit → more burial
            for g in uc.buried_groups:
                if g.get("burial_fraction", 0.0) == 0.0:
                    g["burial_fraction"] = round(burial_est, 2)
    elif uc.buried_groups and uc.binding_mode in ("protein_ligand", "metalloprotein", "protein_ligand_general"):
        for g in uc.buried_groups:
            if g.get("burial_fraction", 0.0) == 0.0:
                g["burial_fraction"] = 0.65

    return uc