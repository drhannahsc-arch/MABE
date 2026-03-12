"""
mabe/glycan/contact_map.py -- Glycan-receptor contact maps
==========================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class OHContact:
    """Contact annotation for a single hydroxyl."""
    position: str
    n_hbonds: int = 0
    hbond_partners: List[str] = field(default_factory=list)
    is_buried: bool = False
    is_solvent_exposed: bool = True


@dataclass
class CHPiContact:
    """CH-pi interaction between pyranose face and aromatic residue."""
    sugar_hydrogens: List[str] = field(default_factory=list)
    receptor_residue: str = ''
    n_CH_contacts: int = 0


@dataclass
class GlycanContactMap:
    """Complete contact map for a glycan-receptor complex."""
    pdb_id: str = ''
    receptor_name: str = ''
    sugar_key: str = ''
    residue_in_binding_site: str = ''

    oh_contacts: List[OHContact] = field(default_factory=list)
    ch_pi_contacts: List[CHPiContact] = field(default_factory=list)
    n_conserved_waters: int = 0
    has_metal: bool = False
    metal_identity: Optional[str] = None
    n_ca_bridges: int = 0

    # G2: Conformational entropy (linkage data)
    linkage_types: List[str] = field(default_factory=list)
    n_branch_points: int = 0

    @property
    def n_hbonds_per_oh(self) -> List[int]:
        return [c.n_hbonds for c in self.oh_contacts]

    @property
    def total_ch_pi(self) -> int:
        return sum(c.n_CH_contacts for c in self.ch_pi_contacts)

    @property
    def positions(self) -> List[str]:
        return [c.position for c in self.oh_contacts]


def cona_mannose_pocket() -> GlycanContactMap:
    """ConA monosaccharide binding pocket. PDB: 5CNA, 1I3H."""
    return GlycanContactMap(
        pdb_id='5CNA', receptor_name='ConA', sugar_key='aMan',
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
        ch_pi_contacts=[],
        n_conserved_waters=1,
        linkage_types=[],
        n_branch_points=0,
    )


def cona_trimannoside() -> GlycanContactMap:
    """ConA trimannoside binding. PDB: 1CVN. Naismith & Field 1996."""
    return GlycanContactMap(
        pdb_id='1CVN', receptor_name='ConA', sugar_key='aMan',
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
