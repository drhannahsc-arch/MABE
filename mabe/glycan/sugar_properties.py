"""
mabe/glycan/sugar_properties.py — Sugar property card generation
================================================================

Generates a SugarPropertyCard for any monosaccharide, containing
all geometric/electronic descriptors needed by the glycan scorer.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class HydroxylDescriptor:
    """Descriptor for a single hydroxyl group."""
    position: str           # 'C1', 'C2', 'C3', 'C4', 'C6'
    orientation: str        # 'axial', 'equatorial', 'primary'
    sasa_key: str           # key into SASA_DESOLV_PER_POSITION
    sasa_desolv_kj: float   # kJ/mol desolvation cost from SASA


@dataclass
class SugarPropertyCard:
    """
    Complete descriptor for a monosaccharide.
    
    Generated once per sugar identity; reused across all binding
    contexts for that sugar.
    """
    name: str                              # 'glucose', 'mannose', 'galactose', etc.
    three_letter: str                      # 'Glc', 'Man', 'Gal', etc.
    anomeric: str                          # 'alpha', 'beta'
    ring_form: str                         # 'pyranose', 'furanose'
    hydroxyls: List[HydroxylDescriptor] = field(default_factory=list)
    n_CH_pi_faces: int = 0                 # axial H count on alpha face
    has_NAc: bool = False                  # N-acetyl group present
    has_carboxylate: bool = False          # e.g., sialic acid
    smiles: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════
# MONOSACCHARIDE LIBRARY
# ═══════════════════════════════════════════════════════════════════════════

def _oh(pos, orient, sasa_key, sasa_val):
    return HydroxylDescriptor(pos, orient, sasa_key, sasa_val)


# α-D-Mannose: C2-axial, all others equatorial
ALPHA_D_MANNOSE = SugarPropertyCard(
    name='mannose', three_letter='Man', anomeric='alpha', ring_form='pyranose',
    hydroxyls=[
        _oh('C1', 'axial',       'C1_anomeric',    3.67),  # anomeric OH
        _oh('C2', 'axial',       'C2_axial',       3.13),  # defining feature of Man
        _oh('C3', 'equatorial',  'C3_equatorial',  3.04),
        _oh('C4', 'equatorial',  'C4_equatorial',  2.48),
        _oh('C6', 'primary',     'C6_primary',     3.40),
    ],
    n_CH_pi_faces=3,  # C1-H, C3-H, C5-H axial on alpha face
)

# α-D-Glucose: all equatorial
ALPHA_D_GLUCOSE = SugarPropertyCard(
    name='glucose', three_letter='Glc', anomeric='alpha', ring_form='pyranose',
    hydroxyls=[
        _oh('C1', 'axial',       'C1_anomeric',    3.67),  # α anomeric
        _oh('C2', 'equatorial',  'C2_equatorial',  3.14),
        _oh('C3', 'equatorial',  'C3_equatorial',  3.04),
        _oh('C4', 'equatorial',  'C4_equatorial',  2.48),
        _oh('C6', 'primary',     'C6_primary',     3.40),
    ],
    n_CH_pi_faces=3,
)

# α-D-Galactose: C4-axial (galacto configuration)
ALPHA_D_GALACTOSE = SugarPropertyCard(
    name='galactose', three_letter='Gal', anomeric='alpha', ring_form='pyranose',
    hydroxyls=[
        _oh('C1', 'axial',       'C1_anomeric',    3.67),
        _oh('C2', 'equatorial',  'C2_equatorial',  3.14),
        _oh('C3', 'equatorial',  'C3_equatorial',  3.04),
        _oh('C4', 'axial',       'C4_axial',       3.11),  # defining feature of Gal
        _oh('C6', 'primary',     'C6_primary',     3.40),
    ],
    n_CH_pi_faces=2,  # fewer axial Hs on alpha face due to C4-ax OH
)

# GlcNAc: glucose with NAc at C2
ALPHA_D_GLCNAC = SugarPropertyCard(
    name='N-acetylglucosamine', three_letter='GlcNAc', anomeric='alpha',
    ring_form='pyranose',
    hydroxyls=[
        _oh('C1', 'axial',       'C1_anomeric',    3.67),
        # C2 has NAc, not free OH
        _oh('C3', 'equatorial',  'C3_equatorial',  3.04),
        _oh('C4', 'equatorial',  'C4_equatorial',  2.48),
        _oh('C6', 'primary',     'C6_primary',     3.40),
    ],
    has_NAc=True,
    n_CH_pi_faces=3,
)

# GalNAc: galactose with NAc at C2
ALPHA_D_GALNAC = SugarPropertyCard(
    name='N-acetylgalactosamine', three_letter='GalNAc', anomeric='alpha',
    ring_form='pyranose',
    hydroxyls=[
        _oh('C1', 'axial',       'C1_anomeric',    3.67),
        # C2 has NAc, not free OH
        _oh('C3', 'equatorial',  'C3_equatorial',  3.04),
        _oh('C4', 'axial',       'C4_axial',       3.11),
        _oh('C6', 'primary',     'C6_primary',     3.40),
    ],
    has_NAc=True,
    n_CH_pi_faces=2,
)

# L-Fucose: 6-deoxy-L-galactose
ALPHA_L_FUCOSE = SugarPropertyCard(
    name='fucose', three_letter='Fuc', anomeric='alpha', ring_form='pyranose',
    hydroxyls=[
        _oh('C1', 'axial',       'C1_anomeric',    3.67),
        _oh('C2', 'equatorial',  'C2_equatorial',  3.14),
        _oh('C3', 'equatorial',  'C3_equatorial',  3.04),
        _oh('C4', 'axial',       'C4_axial',       3.11),
        # No C6-OH (6-deoxy)
    ],
    n_CH_pi_faces=2,
)


SUGAR_LIBRARY = {
    'aMan': ALPHA_D_MANNOSE,
    'aGlc': ALPHA_D_GLUCOSE,
    'aGal': ALPHA_D_GALACTOSE,
    'aGlcNAc': ALPHA_D_GLCNAC,
    'aGalNAc': ALPHA_D_GALNAC,
    'aFuc': ALPHA_L_FUCOSE,
}


def get_sugar_card(sugar_key: str) -> SugarPropertyCard:
    """Look up a sugar property card by key."""
    if sugar_key not in SUGAR_LIBRARY:
        raise KeyError(f"Unknown sugar: {sugar_key}. "
                       f"Available: {list(SUGAR_LIBRARY.keys())}")
    return SUGAR_LIBRARY[sugar_key]
