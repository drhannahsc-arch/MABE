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
    )
