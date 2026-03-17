"""
mabe/glycan/contact_map.py — Glycan-receptor contact maps
==========================================================

Structured representation of sugar-receptor contacts extracted
from crystal structures. Each hydroxyl is annotated with:
  - Number of H-bonds to receptor
  - CH-π contacts (pyranose face to aromatic residues)
  - Whether it's buried or solvent-exposed
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class OHContact:
    """Contact annotation for a single hydroxyl."""
    position: str           # 'C1', 'C2', 'C3', 'C4', 'C6'
    n_hbonds: int = 0       # H-bonds to receptor atoms
    hbond_partners: List[str] = field(default_factory=list)  # e.g. ['Asn14_ND2', 'Leu99_O']
    is_buried: bool = False
    is_solvent_exposed: bool = True


@dataclass
class CHPiContact:
    """CH-π interaction between pyranose face and aromatic residue."""
    sugar_hydrogens: List[str] = field(default_factory=list)  # e.g. ['C1-H', 'C3-H', 'C5-H']
    receptor_residue: str = ''  # e.g. 'Trp181'
    n_CH_contacts: int = 0


@dataclass
class GlycanContactMap:
    """
    Complete contact map for a glycan-receptor complex.
    
    Constructed from crystal structure analysis (PDB + PLIP or manual).
    Consumed by GlycanScorer.
    """
    pdb_id: str = ''
    receptor_name: str = ''
    sugar_key: str = ''           # key into SUGAR_LIBRARY
    residue_in_binding_site: str = ''  # which sugar residue occupies the pocket

    oh_contacts: List[OHContact] = field(default_factory=list)
    ch_pi_contacts: List[CHPiContact] = field(default_factory=list)
    n_conserved_waters: int = 0
    has_metal: bool = False
    metal_identity: Optional[str] = None  # e.g. 'Ca2+' for C-type lectins
    n_ca_bridges: int = 0

    # G2: Conformational entropy (linkage data)
    linkage_types: List[str] = field(default_factory=list)  # e.g. ['alpha1-3', 'alpha1-6']
    n_branch_points: int = 0

    # G3: Detailed CH-pi contacts (optional, for geometric weighting)
    detailed_ch_pi_contacts: Optional[list] = None

    # Derived
    @property
    def n_hbonds_per_oh(self) -> List[int]:
        return [c.n_hbonds for c in self.oh_contacts]

    @property
    def total_ch_pi(self) -> int:
        return sum(c.n_CH_contacts for c in self.ch_pi_contacts)

    @property
    def positions(self) -> List[str]:
        return [c.position for c in self.oh_contacts]


# ═══════════════════════════════════════════════════════════════════════════
# KNOWN CONTACT MAPS (from literature extraction)
# ═══════════════════════════════════════════════════════════════════════════

def cona_mannose_pocket() -> GlycanContactMap:
    """
    ConA monosaccharide binding pocket.
    Sources: Bradbrook 1998, Moothoo & Naismith 1998, Brewer review.
    PDB: 5CNA, 1I3H
    """
    return GlycanContactMap(
        pdb_id='5CNA',
        receptor_name='ConA',
        sugar_key='aMan',
        residue_in_binding_site='alpha(1->6) Man',
        oh_contacts=[
            OHContact('C1', n_hbonds=0, is_solvent_exposed=True),
            OHContact('C2', n_hbonds=0, is_solvent_exposed=True),
            OHContact('C3', n_hbonds=2, hbond_partners=['Asn14_ND2', 'Leu99_O'],
                      is_buried=True, is_solvent_exposed=False),
            OHContact('C4', n_hbonds=2, hbond_partners=['Asp208_OD1', 'Asp208_OD2'],
                      is_buried=True, is_solvent_exposed=False),
            OHContact('C6', n_hbonds=2, hbond_partners=['Asp208_OD', 'Arg228_NH'],
                      is_buried=True, is_solvent_exposed=False),
        ],
        ch_pi_contacts=[],  # ConA has no Trp/Tyr stacking with sugar
        n_conserved_waters=1,  # Asn14-linked water
        # Monosaccharide: no linkages
        linkage_types=[],
        n_branch_points=0,
    )


def cona_trimannoside() -> GlycanContactMap:
    """
    ConA trimannoside binding — alpha(1->6) arm in monosaccharide pocket.
    The trimannoside has alpha1-3 and alpha1-6 linkages with 1 branch point.
    Source: Naismith & Field 1996, Gupta 1996.
    PDB: 1CVN
    """
    return GlycanContactMap(
        pdb_id='1CVN',
        receptor_name='ConA',
        sugar_key='aMan',
        residue_in_binding_site='alpha(1->6) Man (trimannoside)',
        oh_contacts=[
            OHContact('C1', n_hbonds=0, is_solvent_exposed=True),
            OHContact('C2', n_hbonds=0, is_solvent_exposed=True),
            OHContact('C3', n_hbonds=2, hbond_partners=['Asn14_ND2', 'Leu99_O'],
                      is_buried=True, is_solvent_exposed=False),
            OHContact('C4', n_hbonds=2, hbond_partners=['Asp208_OD1', 'Asp208_OD2'],
                      is_buried=True, is_solvent_exposed=False),
            OHContact('C6', n_hbonds=2, hbond_partners=['Asp208_OD', 'Arg228_NH'],
                      is_buried=True, is_solvent_exposed=False),
        ],
        ch_pi_contacts=[],
        n_conserved_waters=1,
        linkage_types=['alpha1-3', 'alpha1-6'],
        n_branch_points=1,
    )


def cona_deoxy_reference() -> GlycanContactMap:
    """ConA pocket for deoxy-sugar prediction validation.

    Same contacts as 5CNA monosaccharide pocket.
    Used with Chervenak & Toone 1995 Biochemistry 34:5685 deoxy-sugar ITC data.
    PDB: 2CNA
    """
    cm = cona_mannose_pocket()
    cm.pdb_id = '2CNA'
    cm.residue_in_binding_site = 'deoxy-sugar validation reference'
    return cm


def galectin3_lacnac() -> GlycanContactMap:
    """Galectin-3 + LacNAc (N-acetyllactosamine).

    Sources: Seetharaman 1998 JBC 273:13047, Fig 3 + Table 2
             Leffler 2004 Glycoconj J 21:433 (review of galectin contacts)
             Diehl 2010 PNAS 107:10299 (Trp181 CH-pi)
    PDB: 3GAL (originally 1A3K also relevant)

    Key: Trp181 stacks on alpha-face of Gal. C4-OH and C6-OH are
    the primary H-bond donors. CH-pi is the dominant selectivity driver.
    """
    return GlycanContactMap(
        pdb_id='3GAL',
        receptor_name='galectin-3',
        sugar_key='LacNAc',
        residue_in_binding_site='Gal(beta1-4)GlcNAc',
        oh_contacts=[
            OHContact('Gal-C4', n_hbonds=2,
                      hbond_partners=['His158_NE2', 'Asn160_ND2'],
                      is_buried=True, is_solvent_exposed=False),
            OHContact('Gal-C6', n_hbonds=2,
                      hbond_partners=['Asn174_ND2', 'Glu184_OE1'],
                      is_buried=True, is_solvent_exposed=False),
            OHContact('Gal-C3', n_hbonds=0, is_solvent_exposed=True),
            OHContact('GlcNAc-C3', n_hbonds=1,
                      hbond_partners=['Arg162_NH2'],
                      is_buried=False, is_solvent_exposed=True),
        ],
        ch_pi_contacts=[
            CHPiContact(
                sugar_hydrogens=['Gal-C3H', 'Gal-C4H', 'Gal-C5H'],
                receptor_residue='Trp181',
                n_CH_contacts=3,
            ),
        ],
        n_conserved_waters=0,
        linkage_types=['beta1-4'],
        n_branch_points=0,
    )


def wga_glcnac() -> GlycanContactMap:
    """WGA (wheat germ agglutinin) + GlcNAc.

    Sources: Wright & Jaeger 1993 JMB 232:620, Table 4
             Wright 1992 JMB 226:1039
    PDB: 2UVO

    Key: WGA has 4 binding sites per monomer (hevein domains).
    Each site: Tyr73 stacks on GlcNAc + Ser H-bonds to OH groups.
    NAc group is critical — GlcNAc >> Glc for WGA.
    """
    return GlycanContactMap(
        pdb_id='2UVO',
        receptor_name='WGA',
        sugar_key='GlcNAc',
        residue_in_binding_site='GlcNAc (primary site)',
        oh_contacts=[
            OHContact('C3', n_hbonds=1, hbond_partners=['Ser62_OG'],
                      is_buried=False, is_solvent_exposed=True),
            OHContact('C6', n_hbonds=1, hbond_partners=['Ser43_OG'],
                      is_buried=False, is_solvent_exposed=True),
            OHContact('NAc-CO', n_hbonds=1, hbond_partners=['backbone_NH'],
                      is_buried=True, is_solvent_exposed=False),
            OHContact('C1', n_hbonds=0, is_solvent_exposed=True),
            OHContact('C4', n_hbonds=0, is_solvent_exposed=True),
        ],
        ch_pi_contacts=[
            CHPiContact(
                sugar_hydrogens=['C1-H', 'C3-H', 'C5-H'],
                receptor_residue='Tyr73',
                n_CH_contacts=3,
            ),
        ],
        n_conserved_waters=1,
        linkage_types=[],
        n_branch_points=0,
    )


def pna_t_antigen() -> GlycanContactMap:
    """PNA (peanut agglutinin) + T-antigen (Gal-beta1,3-GalNAc).

    Sources: Banerjee 1996 PNAS 93:6737, Fig 3
             Ravishankar 1997 Structure 5:1339
    PDB: 2PEL

    Key: PNA recognizes the T-antigen disaccharide.
    No aromatic stacking — binding is H-bond-driven.
    """
    return GlycanContactMap(
        pdb_id='2PEL',
        receptor_name='PNA',
        sugar_key='T-antigen',
        residue_in_binding_site='Gal(beta1-3)GalNAc',
        oh_contacts=[
            OHContact('Gal-C3', n_hbonds=1, hbond_partners=['Asp83_OD'],
                      is_buried=True, is_solvent_exposed=False),
            OHContact('Gal-C4', n_hbonds=2, hbond_partners=['Asp83_OD', 'Gly104_N'],
                      is_buried=True, is_solvent_exposed=False),
            OHContact('Gal-C6', n_hbonds=0, is_solvent_exposed=True),
            OHContact('GalNAc-C6', n_hbonds=1, hbond_partners=['Ser211_OG'],
                      is_buried=False, is_solvent_exposed=True),
        ],
        ch_pi_contacts=[],
        n_conserved_waters=2,
        linkage_types=['beta1-3'],
        n_branch_points=0,
    )


def lysozyme_triNAG() -> GlycanContactMap:
    """Lysozyme + (GlcNAc)3 in subsites A-B-C.

    Sources: Cheetham 1992 JMB 224:613
             Muraki 1997 Biochemistry 36:7695 (Trp62 mutant data)
             Strynadka 1991 JMB 220:401
    PDB: 1LZB

    Key: Trp62 stacks on GlcNAc-C ring face — the primary CH-pi contact.
    Trp62Ala mutation loses ~3 kJ/mol (Muraki 1997) — direct CH-pi validation.
    """
    return GlycanContactMap(
        pdb_id='1LZB',
        receptor_name='lysozyme',
        sugar_key='(GlcNAc)3',
        residue_in_binding_site='GlcNAc-A + GlcNAc-B + GlcNAc-C',
        oh_contacts=[
            OHContact('A-C3', n_hbonds=1, hbond_partners=['Asn59_ND2'],
                      is_buried=True, is_solvent_exposed=False),
            OHContact('A-C6', n_hbonds=1, hbond_partners=['Asp101_OD'],
                      is_buried=False, is_solvent_exposed=True),
            OHContact('B-C3', n_hbonds=1, hbond_partners=['Ala107_O'],
                      is_buried=True, is_solvent_exposed=False),
            OHContact('B-NAc', n_hbonds=1, hbond_partners=['backbone_O'],
                      is_buried=True, is_solvent_exposed=False),
            OHContact('C-C6', n_hbonds=1, hbond_partners=['Asn44_ND2'],
                      is_buried=True, is_solvent_exposed=False),
            OHContact('C-NAc', n_hbonds=1, hbond_partners=['Asn37_OD'],
                      is_buried=True, is_solvent_exposed=False),
        ],
        ch_pi_contacts=[
            CHPiContact(
                sugar_hydrogens=['C-C1H', 'C-C3H', 'C-C5H'],
                receptor_residue='Trp62',
                n_CH_contacts=3,
            ),
            CHPiContact(
                sugar_hydrogens=['A-C1H', 'A-C3H'],
                receptor_residue='Trp63',
                n_CH_contacts=2,
            ),
        ],
        n_conserved_waters=3,
        linkage_types=['beta1-4', 'beta1-4'],
        n_branch_points=0,
    )
