"""
core/porous_descriptor.py — Porous Material Binding Descriptor
===============================================================

Decomposes binding in MOFs, COFs, zeolites, and porous cages into
physics terms that map onto the unified scorer:

  1. Open metal site → metal coordination scorer (same as chelation)
  2. Pore confinement → HG hydrophobic/shape terms (guest-in-cavity)
  3. Linker functional groups → H-bond, π-stacking terms
  4. Framework charge → electrostatic/ion-dipole terms

The framework is NOT described by a single SMILES (extended structure).
Instead: linker SMILES + node type + topology → local binding environment.

Entry point:
  characterize_porous(linker_smiles, node_type, topology, pore_diameter_A,
                      guest_smiles=None) -> dict of UC fields

Sources:
  - Pore diameters: Moghadam et al. Chem. Mater. 29, 2618 (2017)
  - Open metal site energetics: Dinca & Long, JACS 127, 9376 (2005)
  - Confinement effects: Snurr group, JPCL 3, 1159 (2012)
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

_RDKIT_AVAILABLE = False
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski
    _RDKIT_AVAILABLE = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════
# NODE TYPE DATABASE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class NodeType:
    """Metal node / SBU properties."""
    name: str
    formula: str           # e.g. "Zr6O4(OH)4" for UiO-66
    metal: str             # metal formula for scorer: "Zr4+", "Cu2+"
    metal_charge: int
    metal_d_electrons: int
    open_metal_sites: int  # per node, available for guest coordination
    coordination_number: int  # of metal in framework
    node_charge: int       # net charge per node
    geometry: str          # "octahedral", "paddlewheel", "tetrahedral"
    notes: str = ""


NODE_TYPES = {
    # ── Zr-based ──────────────────────────────────────────────────────
    'Zr6': NodeType('Zr6-oxo', 'Zr6O4(OH)4', 'Zr4+', 4, 0,
                    open_metal_sites=0, coordination_number=8,
                    node_charge=0, geometry='octahedral',
                    notes='UiO-66/67/68. No open metal sites in pristine form.'),
    'Zr6-defect': NodeType('Zr6-defect', 'Zr6O4(OH)4', 'Zr4+', 4, 0,
                           open_metal_sites=2, coordination_number=7,
                           node_charge=0, geometry='octahedral',
                           notes='Defective UiO-66. Missing-linker creates OMS.'),

    # ── Cu-based ──────────────────────────────────────────────────────
    'Cu-paddlewheel': NodeType('Cu-paddlewheel', 'Cu2(COO)4', 'Cu2+', 2, 9,
                               open_metal_sites=2, coordination_number=5,
                               node_charge=0, geometry='square_pyramidal',
                               notes='HKUST-1 / MOF-199. Axial OMS per Cu2 unit.'),

    # ── Zn-based ──────────────────────────────────────────────────────
    'Zn4O': NodeType('Zn4O-oxo', 'Zn4O(COO)6', 'Zn2+', 2, 10,
                     open_metal_sites=0, coordination_number=4,
                     node_charge=0, geometry='tetrahedral',
                     notes='MOF-5 / IRMOF series. Saturated Zn.'),

    'Zn2-paddlewheel': NodeType('Zn2-paddlewheel', 'Zn2(COO)4', 'Zn2+', 2, 10,
                                open_metal_sites=2, coordination_number=5,
                                node_charge=0, geometry='square_pyramidal',
                                notes='MOF-2 type.'),

    # ── Fe-based ──────────────────────────────────────────────────────
    'Fe3-oxo': NodeType('Fe3-oxo', 'Fe3O(COO)6', 'Fe3+', 3, 5,
                        open_metal_sites=3, coordination_number=6,
                        node_charge=0, geometry='octahedral',
                        notes='MIL-100/101 type. Each Fe has 1 labile site.'),

    # ── Mg-based ──────────────────────────────────────────────────────
    'Mg-DOBDC': NodeType('Mg-chain', 'Mg2(DOBDC)', 'Mg2+', 2, 0,
                         open_metal_sites=1, coordination_number=5,
                         node_charge=0, geometry='square_pyramidal',
                         notes='MOF-74 / Mg-DOBDC. High-density OMS.'),

    # ── COF (no metal) ───────────────────────────────────────────────
    'organic-imine': NodeType('imine-COF', 'C=N linkage', '', 0, 0,
                              open_metal_sites=0, coordination_number=0,
                              node_charge=0, geometry='planar',
                              notes='Imine-linked COF. No metal sites.'),
    'organic-boronate': NodeType('boronate-COF', 'B-O linkage', '', 0, 0,
                                 open_metal_sites=0, coordination_number=0,
                                 node_charge=0, geometry='planar',
                                 notes='Boronate-ester COF.'),
}


# ═══════════════════════════════════════════════════════════════════════════
# COMMON LINKER PROPERTIES (pre-computed for speed)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LinkerInfo:
    """Properties of a MOF/COF linker relevant to guest binding."""
    name: str
    smiles: str
    n_aromatic_rings: int
    n_hbond_donors: int
    n_hbond_acceptors: int
    has_open_functional_group: bool   # -NH2, -OH, -SH etc. facing pore
    functional_groups: list           # ["NH2", "OH", ...]
    length_A: float                   # end-to-end distance


KNOWN_LINKERS = {
    'BDC': LinkerInfo('1,4-benzenedicarboxylate', 'OC(=O)c1ccc(C(=O)O)cc1',
                      1, 0, 4, False, [], 7.0),
    'NH2-BDC': LinkerInfo('2-amino-1,4-BDC', 'OC(=O)c1cc(N)c(C(=O)O)cc1',
                          1, 1, 5, True, ['NH2'], 7.0),
    'BTC': LinkerInfo('1,3,5-benzenetricarboxylate', 'OC(=O)c1cc(C(=O)O)cc(C(=O)O)c1',
                      1, 0, 6, False, [], 7.0),
    'DOBDC': LinkerInfo('2,5-dihydroxyterephthalic acid', 'OC(=O)c1cc(O)c(C(=O)O)cc1O',
                        1, 2, 6, True, ['OH', 'OH'], 7.0),
    'NDC': LinkerInfo('2,6-naphthalenedicarboxylate', 'OC(=O)c1ccc2cc(C(=O)O)ccc2c1',
                      2, 0, 4, False, [], 9.5),
    'BPDC': LinkerInfo('biphenyl-4,4\'-dicarboxylate', 'OC(=O)c1ccc(-c2ccc(C(=O)O)cc2)cc1',
                       2, 0, 4, False, [], 12.0),
    'TPDC': LinkerInfo('terphenyldicarboxylate', 'OC(=O)c1ccc(-c2ccc(-c3ccc(C(=O)O)cc3)cc2)cc1',
                       3, 0, 4, False, [], 16.5),
    # COF linkers
    'TAPB': LinkerInfo('1,3,5-tris(4-aminophenyl)benzene',
                       'Nc1ccc(-c2cc(-c3ccc(N)cc3)cc(-c3ccc(N)cc3)c2)cc1',
                       4, 3, 3, True, ['NH2', 'NH2', 'NH2'], 12.0),
    'DMTP': LinkerInfo('2,5-dimethoxyterephthalaldehyde', 'COc1cc(C=O)c(OC)cc1C=O',
                       1, 0, 4, False, [], 7.0),
}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def characterize_porous(
    linker_smiles: str = "",
    node_type: str = "",
    topology: str = "",
    pore_diameter_A: float = 0.0,
    surface_area_m2g: float = 0.0,
    guest_smiles: str = "",
    linker_name: str = "",
    framework_name: str = "",
) -> Dict:
    """
    Characterize a porous material binding site.

    Decomposes into:
      - Metal coordination (open metal sites)
      - Pore confinement (cavity/pore geometry)
      - Linker interactions (H-bond, π)

    Returns dict of UniversalComplex field values.
    """
    result = {
        'host_name': framework_name or f"{node_type}+{linker_name or 'linker'}",
        'host_type': 'porous_framework',
        'binding_mode': 'porous_framework',
    }

    # ── Node properties ───────────────────────────────────────────────
    node = NODE_TYPES.get(node_type)
    if node is not None:
        result['metal_formula'] = node.metal
        result['metal_charge'] = node.metal_charge
        result['metal_d_electrons'] = node.metal_d_electrons
        result['geometry'] = node.geometry

        # Open metal sites → donor subtypes for metal scorer
        if node.open_metal_sites > 0 and guest_smiles:
            # The guest coordinates to the OMS
            # Donor subtype depends on what guest atom binds
            donor_subtypes = _guest_donors_for_oms(guest_smiles, node)
            result['donor_subtypes'] = donor_subtypes
            result['denticity'] = len(donor_subtypes)
    else:
        # No metal node (COF or unknown)
        result['metal_formula'] = ''

    # ── Pore as cavity ────────────────────────────────────────────────
    if pore_diameter_A > 0:
        pore_radius_A = pore_diameter_A / 2
        # Model pore as cylinder segment of length = pore_diameter (local environment)
        # Effective cavity volume ≈ π r² × r (one diameter length)
        cav_vol = math.pi * pore_radius_A**2 * pore_radius_A
        result['cavity_volume_A3'] = round(cav_vol, 1)
        result['cavity_radius_nm'] = round(pore_radius_A / 10, 4)
    else:
        # Estimate from linker length
        linker_info = KNOWN_LINKERS.get(linker_name)
        if linker_info:
            # Pore diameter ≈ linker length (rough for cubic topology)
            pore_d = linker_info.length_A
            pore_r = pore_d / 2
            cav_vol = math.pi * pore_r**2 * pore_r
            result['cavity_volume_A3'] = round(cav_vol, 1)
            result['cavity_radius_nm'] = round(pore_r / 10, 4)

    # ── Linker functional groups ──────────────────────────────────────
    linker_info = KNOWN_LINKERS.get(linker_name)
    if linker_info is not None:
        result['n_aromatic_walls'] = linker_info.n_aromatic_rings
        result['n_hbond_donors_host'] = linker_info.n_hbond_donors
        result['n_hbond_acceptors_host'] = linker_info.n_hbond_acceptors
    elif linker_smiles and _RDKIT_AVAILABLE:
        mol = Chem.MolFromSmiles(linker_smiles)
        if mol:
            result['n_aromatic_walls'] = rdMolDescriptors.CalcNumAromaticRings(mol)
            result['n_hbond_donors_host'] = Lipinski.NumHDonors(mol)
            result['n_hbond_acceptors_host'] = Lipinski.NumHAcceptors(mol)

    result['is_macrocyclic'] = False  # Porous frameworks aren't macrocycles
    result['is_cage'] = True  # Pore confinement = cage-like

    # ── Guest properties ──────────────────────────────────────────────
    if guest_smiles and _RDKIT_AVAILABLE:
        from core.guest_compute import compute_guest_properties, estimate_sasa_burial
        guest_props = compute_guest_properties(guest_smiles)
        result.update(guest_props)
        result['guest_smiles'] = guest_smiles

        # Packing coefficient
        guest_vol = guest_props.get('guest_volume_A3', 0)
        cav_vol = result.get('cavity_volume_A3', 0)
        if cav_vol > 0 and guest_vol > 0:
            result['packing_coefficient'] = round(guest_vol / cav_vol, 3)

        # H-bond estimation: guest donors to linker acceptors + vice versa
        guest_hbd = guest_props.get('guest_n_hbond_donors', 0)
        guest_hba = guest_props.get('guest_n_hbond_acceptors', 0)
        host_hbd = result.get('n_hbond_donors_host', 0)
        host_hba = result.get('n_hbond_acceptors_host', 0)
        result['n_hbonds_formed'] = min(host_hbd, guest_hba) + min(host_hba, guest_hbd)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# GUEST DONOR ASSIGNMENT FOR OPEN METAL SITES
# ═══════════════════════════════════════════════════════════════════════════

def _guest_donors_for_oms(guest_smiles: str, node: NodeType) -> List[str]:
    """
    Determine what donor subtypes a guest molecule presents to an open metal site.

    For small gas molecules: well-defined.
    For larger molecules: use SMARTS-based donor extraction (same as auto_descriptor).
    """
    # Quick lookup for common gas/small-molecule guests
    COMMON_GUESTS = {
        'O=C=O': ['O_carbonyl'],         # CO2 → binds via O to OMS
        'O': ['O_water'],                 # H2O
        '[H][H]': [],                     # H2 → no strong coordination
        'N#N': ['N_nitrile'],             # N2 → weak
        'C=O': ['O_carbonyl'],            # CO → via C or O
        'O=S=O': ['O_sulfonyl'],          # SO2
        '[NH3]': ['N_amine'],             # NH3
        'CC=O': ['O_carbonyl'],           # acetaldehyde
        'CO': ['O_hydroxyl'],             # methanol
        'CCO': ['O_hydroxyl'],            # ethanol
        'OC(=O)O': ['O_carboxylate'],     # carbonic acid
    }

    if guest_smiles in COMMON_GUESTS:
        return COMMON_GUESTS[guest_smiles]

    # For complex guests: extract from SMARTS
    if not _RDKIT_AVAILABLE:
        return ['O_generic']

    mol = Chem.MolFromSmiles(guest_smiles)
    if mol is None:
        return []

    subtypes = []
    # Priority order: most coordinating first
    patterns = [
        ('O_carboxylate', '[OX1]C(=O)'),
        ('N_amine', '[NX3;H2,H1]'),
        ('O_hydroxyl', '[OX2H]'),
        ('N_imidazole', 'c1cn[nH]c1'),
        ('N_pyridine', 'n'),
        ('O_carbonyl', '[CX3]=O'),
        ('S_thiolate', '[SX1,SX2H]'),
    ]

    for subtype, smarts in patterns:
        pat = Chem.MolFromSmarts(smarts)
        if pat and mol.HasSubstructMatch(pat):
            n_matches = len(mol.GetSubstructMatches(pat))
            # One donor per OMS
            for _ in range(min(n_matches, node.open_metal_sites)):
                subtypes.append(subtype)
            if len(subtypes) >= node.open_metal_sites:
                break

    return subtypes[:node.open_metal_sites]


# ═══════════════════════════════════════════════════════════════════════════
# TOPOLOGY DATABASE (for pore size estimation)
# ═══════════════════════════════════════════════════════════════════════════

TOPOLOGY_PORE_FACTORS = {
    # topology_name: pore_diameter / linker_length ratio
    'pcu': 1.0,   # primitive cubic (MOF-5, IRMOF series)
    'fcu': 0.7,   # face-centered cubic (UiO-66)
    'tbo': 0.9,   # HKUST-1 type
    'acs': 0.8,   # MOF-74 type (1D channels)
    'hcb': 1.2,   # honeycomb (2D COFs)
    'sql': 1.1,   # square lattice
    'dia': 0.6,   # diamond (interpenetrated)
}
