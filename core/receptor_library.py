"""
core/receptor_library.py — Scaffold Libraries for De Novo Design
=================================================================

Two libraries:
  A. Synthetic receptors: urea cages, molecular tweezers, macrocycles,
     expanded calixarenes, pillar[n]arenes, crown-amide hybrids
  B. Porous material building blocks: MOF linkers, COF monomers,
     SBU types, topology specifications

Used by de_novo_generator for combinatorial enumeration
and by design_engine_v2 for targeted scaffold selection.

Convention: [*] marks attachment points for arm/functional group grafting.
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ═══════════════════════════════════════════════════════════════════════════
# SYNTHETIC RECEPTOR SCAFFOLDS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ReceptorScaffold:
    """A host scaffold for combinatorial design."""
    name: str
    smiles: str              # SMILES with [*] attachment points
    n_sites: int             # Number of derivatizable positions
    scaffold_type: str       # 'cage', 'macrocycle', 'tweezer', 'clip', 'foldamer'
    cavity_class: str        # 'small' (<100 A3), 'medium' (100-300), 'large' (>300)
    n_aromatic_walls: int
    n_hbond_sites: int       # Intrinsic H-bond donors + acceptors
    binding_modalities: list  # ['hydrophobic', 'hbond', 'ch_pi', 'cation_pi', 'anion']
    reference: str = ""
    notes: str = ""


RECEPTOR_SCAFFOLDS = [
    # ── UREA CAGES (Davis-type glucose receptors) ─────────────────────
    ReceptorScaffold(
        'triurea-cage-2site', '[*]NC(=O)Nc1ccc(NC(=O)Nc2ccc(NC(=O)N[*])cc2)cc1',
        n_sites=2, scaffold_type='cage', cavity_class='medium',
        n_aromatic_walls=2, n_hbond_sites=12,
        binding_modalities=['hbond', 'ch_pi'],
        reference='Tromans 2019 Nature Chem. 11:52',
        notes='Davis GluHUT-type cage. H-bond array for polyol recognition.'),

    ReceptorScaffold(
        'bisurea-macrocycle', '[*]NC(=O)NCC[*]',
        n_sites=2, scaffold_type='macrocycle', cavity_class='small',
        n_aromatic_walls=0, n_hbond_sites=4,
        binding_modalities=['hbond'],
        notes='Simple bisurea macrocycle. Size-tunable.'),

    # ── MOLECULAR TWEEZERS ────────────────────────────────────────────
    ReceptorScaffold(
        'klamer-tweezer', '[*]c1cc2ccccc2c2ccccc12',
        n_sites=1, scaffold_type='tweezer', cavity_class='medium',
        n_aromatic_walls=4, n_hbond_sites=0,
        binding_modalities=['ch_pi', 'hydrophobic', 'cation_pi'],
        reference='Klamer group, Angew. Chem.',
        notes='Naphthalene sidewalls for aromatic guest inclusion.'),

    ReceptorScaffold(
        'phosphate-tweezer', '[*]c1cc2cc(P(=O)([O-])[O-])cc3cc(c1)c1cc(cc(c1)P(=O)([O-])[O-])c23',
        n_sites=1, scaffold_type='tweezer', cavity_class='medium',
        n_aromatic_walls=4, n_hbond_sites=4,
        binding_modalities=['cation_pi', 'electrostatic', 'ch_pi'],
        reference='Klamer CLR01/CLR03',
        notes='Anionic tweezers for Lys/Arg recognition.'),

    # ── MACROCYCLIC RECEPTORS ─────────────────────────────────────────
    ReceptorScaffold(
        'amide-macrocycle-4site', '[*]NC(=O)c1ccc([*])cc1',
        n_sites=2, scaffold_type='macrocycle', cavity_class='medium',
        n_aromatic_walls=1, n_hbond_sites=2,
        binding_modalities=['hbond', 'ch_pi'],
        notes='Aromatic amide macrocycle. Preorganized H-bond array.'),

    ReceptorScaffold(
        'crown-amide-hybrid', '[*]NC(=O)COCCOCCO[*]',
        n_sites=2, scaffold_type='macrocycle', cavity_class='medium',
        n_aromatic_walls=0, n_hbond_sites=5,
        binding_modalities=['hbond', 'cation'],
        notes='Crown ether + amide. Binds cations + neutral guests.'),

    # ── EXPANDED CALIXARENES ──────────────────────────────────────────
    ReceptorScaffold(
        'calix4-upper-rim', '[*]c1cc(CC2CC(c3cc([*])cc(CC4CC(c5cc([*])cc(C)c5O)CC(c5cc([*])cc(C)c5O)C4)c3O)CC(c3cc(C)cc(C)c3O)C2)cc(C)c1O',
        n_sites=4, scaffold_type='macrocycle', cavity_class='medium',
        n_aromatic_walls=4, n_hbond_sites=4,
        binding_modalities=['cation_pi', 'ch_pi', 'hydrophobic'],
        notes='Calix[4]arene upper rim functionalization.'),

    ReceptorScaffold(
        'resorcinarene', '[*]c1cc(O)c(CC2(CC)c3cc(O)c([*])cc3)cc1',
        n_sites=2, scaffold_type='macrocycle', cavity_class='large',
        n_aromatic_walls=4, n_hbond_sites=8,
        binding_modalities=['hbond', 'ch_pi', 'hydrophobic'],
        notes='Resorcin[4]arene deep cavity.'),

    # ── PILLAR[n]ARENE VARIANTS ───────────────────────────────────────
    ReceptorScaffold(
        'pillar5-functionalized', '[*]COc1cc(COC)cc(Cc2cc(COC)cc(C[*])c2OC)c1OC',
        n_sites=2, scaffold_type='macrocycle', cavity_class='small',
        n_aromatic_walls=5, n_hbond_sites=0,
        binding_modalities=['ch_pi', 'hydrophobic'],
        notes='Pillar[5]arene with derivatizable rim.'),

    # ── FOLDAMERS ─────────────────────────────────────────────────────
    ReceptorScaffold(
        'aromatic-foldamer-2site', '[*]c1ccnc(-c2cccc(-c3ccnc([*])c3)c2)c1',
        n_sites=2, scaffold_type='foldamer', cavity_class='medium',
        n_aromatic_walls=3, n_hbond_sites=2,
        binding_modalities=['ch_pi', 'hbond'],
        notes='Pyridine-phenyl foldamer helix. Tunable cavity.'),
]


# ═══════════════════════════════════════════════════════════════════════════
# FUNCTIONAL GROUP ARMS (for receptor derivatization)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FunctionalArm:
    """A functional group fragment for receptor modification."""
    name: str
    smiles: str              # SMILES with [*] attachment
    group_type: str          # 'hbond_donor', 'hbond_acceptor', 'aromatic',
                             # 'charged', 'hydrophobic', 'metal_binding'
    interaction_type: str    # Maps to scorer term
    notes: str = ""


RECEPTOR_ARMS = [
    # ── H-bond donors ─────────────────────────────────────────────────
    FunctionalArm('urea', '[*]NC(=O)N', 'hbond_donor', 'hbond',
                  'Strong directional H-bond donor pair'),
    FunctionalArm('squaramide', '[*]NC1=C(N)C(=O)C1=O', 'hbond_donor', 'hbond',
                  'Stronger than urea, more acidic NH'),
    FunctionalArm('amide', '[*]NC(=O)C', 'hbond_donor', 'hbond'),
    FunctionalArm('hydroxyl', '[*]O', 'hbond_donor', 'hbond'),
    FunctionalArm('amino', '[*]N', 'hbond_donor', 'hbond'),
    FunctionalArm('guanidinium', '[*]NC(=N)N', 'charged', 'salt_bridge',
                  'Oxoanion recognition'),

    # ── H-bond acceptors ──────────────────────────────────────────────
    FunctionalArm('pyridine', '[*]c1ccncc1', 'hbond_acceptor', 'hbond'),
    FunctionalArm('carbonyl', '[*]C(=O)C', 'hbond_acceptor', 'hbond'),
    FunctionalArm('ether', '[*]COCC', 'hbond_acceptor', 'hbond'),

    # ── Aromatic walls ────────────────────────────────────────────────
    FunctionalArm('phenyl', '[*]c1ccccc1', 'aromatic', 'ch_pi'),
    FunctionalArm('naphthyl', '[*]c1ccc2ccccc2c1', 'aromatic', 'ch_pi',
                  'Larger π surface for CH-π'),
    FunctionalArm('pyrene', '[*]c1cc2ccc3cccc4ccc(c1)c2c34', 'aromatic', 'ch_pi',
                  'Maximum π surface'),
    FunctionalArm('indole', '[*]c1c[nH]c2ccccc12', 'aromatic', 'ch_pi',
                  'Trp-like CH-π donor'),

    # ── Charged groups ────────────────────────────────────────────────
    FunctionalArm('sulfonate', '[*]S(=O)(=O)[O-]', 'charged', 'electrostatic',
                  'Water-solubilizing, cation binding'),
    FunctionalArm('phosphonate', '[*]P(=O)([O-])[O-]', 'charged', 'electrostatic'),
    FunctionalArm('carboxylate', '[*]C(=O)[O-]', 'charged', 'electrostatic'),
    FunctionalArm('quaternary-N', '[*]C[N+](C)(C)C', 'charged', 'electrostatic',
                  'Permanent cation, anion binding'),

    # ── Hydrophobic ───────────────────────────────────────────────────
    FunctionalArm('tert-butyl', '[*]C(C)(C)C', 'hydrophobic', 'hydrophobic'),
    FunctionalArm('adamantyl', '[*]C12CC3CC(CC(C3)C1)C2', 'hydrophobic', 'hydrophobic'),
    FunctionalArm('cyclohexyl', '[*]C1CCCCC1', 'hydrophobic', 'hydrophobic'),

    # ── Metal-binding ─────────────────────────────────────────────────
    FunctionalArm('catechol', '[*]c1ccc(O)c(O)c1', 'metal_binding', 'metal',
                  'Bidentate for Fe3+, hard metals'),
    FunctionalArm('bipyridyl', '[*]c1ccnc(-c2ccccn2)c1', 'metal_binding', 'metal',
                  'Bidentate N-donor for soft metals'),
    FunctionalArm('hydroxamic', '[*]C(=O)NO', 'metal_binding', 'metal',
                  'Strong Fe3+ binder (siderophore motif)'),

    # ── Caps ──────────────────────────────────────────────────────────
    FunctionalArm('H-cap', '[*][H]', 'cap', 'none', 'Terminate unused site'),
    FunctionalArm('methyl-cap', '[*]C', 'cap', 'none'),
]


# ═══════════════════════════════════════════════════════════════════════════
# POROUS MATERIAL BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MOFLinker:
    """A MOF/COF linker for framework construction."""
    name: str
    smiles: str
    connectivity: int        # 2=ditopic, 3=tritopic, 4=tetratopic
    length_class: str        # 'short' (<8A), 'medium' (8-14A), 'long' (>14A)
    functional_groups: list  # Available for post-synthetic modification
    compatible_nodes: list   # Node types that work with this linker
    reference: str = ""


MOF_LINKERS = [
    MOFLinker('BDC', 'OC(=O)c1ccc(C(=O)O)cc1', 2, 'short',
             [], ['Zr6', 'Cu-paddlewheel', 'Zn4O', 'Fe3-oxo'],
             'MOF-5, UiO-66, HKUST-1'),
    MOFLinker('NH2-BDC', 'OC(=O)c1cc(N)c(C(=O)O)cc1', 2, 'short',
             ['NH2'], ['Zr6', 'Fe3-oxo'],
             'UiO-66-NH2, MIL-101-NH2'),
    MOFLinker('DOBDC', 'OC(=O)c1cc(O)c(C(=O)O)cc1O', 2, 'short',
             ['OH', 'OH'], ['Mg-DOBDC'],
             'MOF-74 series'),
    MOFLinker('BTC', 'OC(=O)c1cc(C(=O)O)cc(C(=O)O)c1', 3, 'short',
             [], ['Cu-paddlewheel', 'Fe3-oxo'],
             'HKUST-1, MIL-100'),
    MOFLinker('NDC', 'OC(=O)c1ccc2cc(C(=O)O)ccc2c1', 2, 'medium',
             [], ['Zr6', 'Zn4O'],
             'DUT-52'),
    MOFLinker('BPDC', 'OC(=O)c1ccc(-c2ccc(C(=O)O)cc2)cc1', 2, 'medium',
             [], ['Zr6', 'Zn4O'],
             'UiO-67, MOF-508'),
    MOFLinker('TPDC', 'OC(=O)c1ccc(-c2ccc(-c3ccc(C(=O)O)cc3)cc2)cc1', 2, 'long',
             [], ['Zr6'],
             'UiO-68'),

    # COF building blocks
    MOFLinker('TAPB', 'Nc1ccc(-c2cc(-c3ccc(N)cc3)cc(-c3ccc(N)cc3)c2)cc1', 3, 'medium',
             ['NH2', 'NH2', 'NH2'], ['organic-imine'],
             'Imine COF amine node'),
    MOFLinker('TFP', 'O=Cc1cc(C=O)cc(C=O)c1', 3, 'short',
             ['CHO', 'CHO', 'CHO'], ['organic-imine'],
             'Trialdehyde COF linker'),
]


# ═══════════════════════════════════════════════════════════════════════════
# LOOKUP HELPERS
# ═══════════════════════════════════════════════════════════════════════════

RECEPTOR_SCAFFOLD_BY_NAME = {s.name: s for s in RECEPTOR_SCAFFOLDS}
RECEPTOR_ARM_BY_NAME = {a.name: a for a in RECEPTOR_ARMS}
MOF_LINKER_BY_NAME = {l.name: l for l in MOF_LINKERS}


def scaffolds_for_target(target_type: str) -> List[ReceptorScaffold]:
    """Return scaffolds suitable for a given target type."""
    if target_type in ('sugar', 'polyol', 'carbohydrate'):
        return [s for s in RECEPTOR_SCAFFOLDS
                if 'hbond' in s.binding_modalities]
    elif target_type in ('cation', 'ammonium', 'lysine'):
        return [s for s in RECEPTOR_SCAFFOLDS
                if 'cation_pi' in s.binding_modalities or 'electrostatic' in s.binding_modalities]
    elif target_type in ('aromatic', 'drug', 'hydrophobic'):
        return [s for s in RECEPTOR_SCAFFOLDS
                if 'hydrophobic' in s.binding_modalities or 'ch_pi' in s.binding_modalities]
    elif target_type in ('anion', 'phosphate', 'sulfate'):
        return [s for s in RECEPTOR_SCAFFOLDS
                if 'hbond' in s.binding_modalities]  # Urea/squaramide for anions
    else:
        return RECEPTOR_SCAFFOLDS  # All


def linkers_for_node(node_type: str) -> List[MOFLinker]:
    """Return MOF linkers compatible with a given node type."""
    return [l for l in MOF_LINKERS if node_type in l.compatible_nodes]
