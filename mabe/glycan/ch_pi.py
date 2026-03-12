"""
mabe/glycan/ch_pi.py -- G3: CH-pi Stacking Interactions
=========================================================

Computes CH-pi stabilization between pyranose aliphatic faces
and aromatic receptor residues (Trp, Tyr, Phe, His).

Physics:
  The apolar face of pyranose rings presents axial C-H bonds that
  stack against aromatic pi-systems. This is the dominant selectivity
  mechanism in many lectins (WGA, hevein, galectins).

  dG_CH_pi = n_contacts * eps_CH_pi * f_geometry * f_aromatic

Parameter Sources (zero biology):
  - eps_CH_pi = -2.5 kJ/mol per contact: Laughrey et al. 2008
    (JACS 130:14625) — synthetic aromatic host + cyclohexane in water
  - Geometric weighting: Nishio 2011 (PCCP 13:13873) — CSD statistics
    preferred distance 3.2-3.8 A, angular preference for C-H...centroid
  - Aromatic hierarchy: Tsuzuki 2000 (JACS 122:3746), Gung 2006
    (J. Org. Chem. 71:9261) — indole > phenol > benzene > imidazole
  - Face selectivity: Asensio 2013 (Acc. Chem. Res. 46:946)
    axial C-H presentation rules per sugar/anomer

Cross-check:
  Hevein double-Trp stacking = 6.3-8.4 kJ/mol total (Asensio 2013)
  = 2-3 contacts x 2.5 kJ/mol + aromatic bonus. Consistent.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


# =====================================================================
# CORE PARAMETER (LOCKED)
# =====================================================================

EPS_CH_PI = -2.5  # kJ/mol per CH-pi contact
# Source: Laughrey et al. 2008 (JACS 130:14625)
# Range from literature: -2.1 to -3.3 kJ/mol
# Locked at -2.5 (median of Laughrey + Asensio glycopeptide data)


# =====================================================================
# AROMATIC RESIDUE HIERARCHY
# =====================================================================
# Relative CH-pi strength by aromatic identity.
# Normalized so Trp (indole) = 1.0.
# Source: Tsuzuki CCSD(T) benchmarks + Gung torsion balance experiments.
# indole > phenol > benzene > imidazole
# This scales eps_CH_pi by aromatic identity.

AROMATIC_WEIGHT = {
    'Trp': 1.00,    # indole: largest pi surface, strongest
    'Tyr': 0.85,    # phenol: slightly weaker due to OH
    'Phe': 0.80,    # benzene: no heteroatom polarization
    'His': 0.55,    # imidazole: small, partially charged at neutral pH
}


# =====================================================================
# GEOMETRIC WEIGHTING
# =====================================================================
# CH-pi energy depends on distance and angle.
# Optimal geometry from Nishio CSD statistics:
#   distance (C-H to ring centroid): 3.2-3.8 A optimal
#   angle (C-H...centroid): near perpendicular preferred
#
# We use a Gaussian distance weighting centered at 3.5 A
# with sigma = 0.5 A (drops to 50% at 2.8 and 4.2 A).
# Contacts beyond 4.5 A are negligible.

D_OPTIMAL = 3.5    # A, center of distance weighting
D_SIGMA = 0.5      # A, width of Gaussian
D_CUTOFF = 4.5     # A, beyond this = no contact

# Angle weighting: perpendicular (90 deg from ring plane) is optimal.
# Modeled as cos^2(theta) where theta is deviation from perpendicular.
# At 0 deg deviation (perpendicular): weight = 1.0
# At 45 deg deviation: weight = 0.5
# At 90 deg deviation (parallel): weight = 0.0


def f_distance(d_angstrom: float) -> float:
    """
    Gaussian distance weighting for CH-pi contact.
    Returns 0-1 weight based on C-H to ring centroid distance.
    """
    if d_angstrom > D_CUTOFF or d_angstrom < 2.0:
        return 0.0
    return math.exp(-0.5 * ((d_angstrom - D_OPTIMAL) / D_SIGMA) ** 2)


def f_angle(deviation_deg: float) -> float:
    """
    Angular weighting for CH-pi contact.
    deviation_deg = 0 means perpendicular to ring (optimal).
    deviation_deg = 90 means parallel to ring (no contact).
    """
    if deviation_deg >= 90 or deviation_deg < 0:
        return 0.0
    rad = math.radians(deviation_deg)
    return math.cos(rad) ** 2


# =====================================================================
# SUGAR FACE CH-pi PRESENTATION
# =====================================================================
# Which axial C-H bonds are presented on each face of the pyranose.
# The alpha-face (below the ring) and beta-face (above) present
# different C-H patterns depending on hydroxyl stereochemistry.
#
# Source: Asensio 2013 (Acc. Chem. Res. 46:946), Figure 2.
# Rule: faces with axial C-H bonds stack; faces with axial OH repel.

@dataclass
class PyranoseFace:
    """CH-pi presentation of one face of a pyranose ring."""
    face: str                    # 'alpha' or 'beta'
    axial_CH_positions: List[str]   # e.g. ['C1-H', 'C3-H', 'C5-H']
    n_axial_CH: int
    stackable: bool              # True if face can form CH-pi contacts
    notes: str = ''


@dataclass
class SugarCHPiProfile:
    """Complete CH-pi profile for a monosaccharide."""
    sugar_key: str
    alpha_face: PyranoseFace
    beta_face: PyranoseFace
    max_simultaneous_contacts: int   # max CH-pi from a single aromatic
    sandwich_possible: bool          # can two aromatics stack both faces?

    @property
    def best_face(self) -> PyranoseFace:
        """Face with most axial C-H bonds (preferred stacking face)."""
        if self.alpha_face.n_axial_CH >= self.beta_face.n_axial_CH:
            return self.alpha_face
        return self.beta_face


# =====================================================================
# SUGAR FACE LIBRARY
# =====================================================================
# From Asensio 2013 systematic analysis.

# alpha-D-Mannose: C2-ax OH blocks beta face
CH_PI_PROFILES = {
    'aMan': SugarCHPiProfile(
        sugar_key='aMan',
        alpha_face=PyranoseFace('alpha', ['C1-H', 'C3-H', 'C5-H'], 3, True,
                                'Standard alpha-face, 3 axial CH'),
        beta_face=PyranoseFace('beta', [], 0, False,
                               'C2-ax OH blocks beta face stacking'),
        max_simultaneous_contacts=3,
        sandwich_possible=False,
    ),
    'bMan': SugarCHPiProfile(
        sugar_key='bMan',
        alpha_face=PyranoseFace('alpha', ['C3-H', 'C5-H'], 2, True,
                                'beta-Man: C1-H now equatorial'),
        beta_face=PyranoseFace('beta', [], 0, False, 'C2-ax OH blocks'),
        max_simultaneous_contacts=2,
        sandwich_possible=False,
    ),
    'aGlc': SugarCHPiProfile(
        sugar_key='aGlc',
        alpha_face=PyranoseFace('alpha', ['C1-H', 'C3-H', 'C5-H'], 3, True),
        beta_face=PyranoseFace('beta', ['C2-H', 'C4-H'], 2, True,
                               'All-equatorial: both faces accessible'),
        max_simultaneous_contacts=3,
        sandwich_possible=False,  # alpha-Glc: alpha face only (Asensio rule)
    ),
    'bGlc': SugarCHPiProfile(
        sugar_key='bGlc',
        alpha_face=PyranoseFace('alpha', ['C3-H', 'C5-H'], 2, True),
        beta_face=PyranoseFace('beta', ['C1-H', 'C2-H', 'C4-H'], 3, True,
                               'beta-Glc: both faces stack (sandwich possible)'),
        max_simultaneous_contacts=3,
        sandwich_possible=True,
    ),
    'aGal': SugarCHPiProfile(
        sugar_key='aGal',
        alpha_face=PyranoseFace('alpha', ['C1-H', 'C3-H', 'C5-H'], 3, True,
                                'C4-ax OH on beta face, alpha face clear'),
        beta_face=PyranoseFace('beta', [], 0, False,
                               'C4-axial OH blocks beta face'),
        max_simultaneous_contacts=3,
        sandwich_possible=False,
    ),
    'bGal': SugarCHPiProfile(
        sugar_key='bGal',
        alpha_face=PyranoseFace('alpha', ['C3-H', 'C5-H'], 2, True),
        beta_face=PyranoseFace('beta', ['C1-H'], 1, True,
                               'beta-Gal: weak beta face'),
        max_simultaneous_contacts=2,
        sandwich_possible=False,
    ),
    'aGlcNAc': SugarCHPiProfile(
        sugar_key='aGlcNAc',
        alpha_face=PyranoseFace('alpha', ['C1-H', 'C3-H', 'C5-H'], 3, True,
                                'Same as Glc + NAc methyl adds CH3 contacts'),
        beta_face=PyranoseFace('beta', ['C2-H', 'C4-H'], 2, True),
        max_simultaneous_contacts=3,  # can be higher with NAc methyl
        sandwich_possible=False,
    ),
    'bGlcNAc': SugarCHPiProfile(
        sugar_key='bGlcNAc',
        alpha_face=PyranoseFace('alpha', ['C3-H', 'C5-H'], 2, True),
        beta_face=PyranoseFace('beta', ['C1-H', 'C2-H', 'C4-H'], 3, True,
                               'beta-GlcNAc: sandwich possible (WGA, hevein)'),
        max_simultaneous_contacts=3,
        sandwich_possible=True,
    ),
    'aFuc': SugarCHPiProfile(
        sugar_key='aFuc',
        alpha_face=PyranoseFace('alpha', ['C1-H', 'C3-H', 'C5-H'], 3, True,
                                'alpha-L-Fuc: alpha face like Gal'),
        beta_face=PyranoseFace('beta', [], 0, False, 'C4-ax OH blocks'),
        max_simultaneous_contacts=3,
        sandwich_possible=False,
    ),
}


def get_ch_pi_profile(sugar_key: str) -> Optional[SugarCHPiProfile]:
    """Look up CH-pi profile. Returns None if not in library."""
    return CH_PI_PROFILES.get(sugar_key)


# =====================================================================
# SCORING FUNCTION
# =====================================================================

@dataclass
class CHPiContact:
    """A single CH-pi contact for scoring."""
    sugar_position: str      # e.g. 'C3-H'
    aromatic_residue: str    # e.g. 'Trp181'
    aromatic_type: str       # 'Trp', 'Tyr', 'Phe', 'His'
    distance_A: Optional[float] = None     # C-H to centroid distance
    angle_deviation_deg: Optional[float] = None  # deviation from perpendicular


def score_ch_pi(
    contacts: List[CHPiContact],
    eps: float = EPS_CH_PI,
    use_geometry: bool = True,
) -> Dict[str, float]:
    """
    Score CH-pi interactions from a contact list.

    Parameters
    ----------
    contacts : list of CHPiContact
        Each contact has sugar position, aromatic identity,
        and optional distance/angle for geometric weighting.
    eps : float
        Energy per ideal contact (kJ/mol). Default: -2.5 (LOCKED).
    use_geometry : bool
        If True and distance/angle are provided, apply geometric weighting.
        If False, count contacts with aromatic scaling only.

    Returns
    -------
    dict with:
        'dG_ch_pi': total CH-pi energy (kJ/mol)
        'n_contacts': number of contacts
        'per_contact': list of per-contact energies
        'aromatic_breakdown': dict of {residue: energy}
    """
    if not contacts:
        return {
            'dG_ch_pi': 0.0,
            'n_contacts': 0,
            'per_contact': [],
            'aromatic_breakdown': {},
        }

    per_contact = []
    aromatic_breakdown = {}

    for c in contacts:
        # Aromatic scaling
        arom_weight = AROMATIC_WEIGHT.get(c.aromatic_type, 0.75)

        # Geometric weighting
        if use_geometry and c.distance_A is not None:
            geo_weight = f_distance(c.distance_A)
            if c.angle_deviation_deg is not None:
                geo_weight *= f_angle(c.angle_deviation_deg)
        else:
            geo_weight = 1.0  # assume ideal geometry if not specified

        energy = eps * arom_weight * geo_weight
        per_contact.append(energy)

        # Accumulate per residue
        res = c.aromatic_residue
        aromatic_breakdown[res] = aromatic_breakdown.get(res, 0.0) + energy

    return {
        'dG_ch_pi': round(sum(per_contact), 3),
        'n_contacts': len(contacts),
        'per_contact': [round(e, 3) for e in per_contact],
        'aromatic_breakdown': {k: round(v, 3) for k, v in aromatic_breakdown.items()},
    }


# =====================================================================
# CONVENIENCE: estimate contacts from sugar profile + receptor aromatics
# =====================================================================

def estimate_ch_pi_contacts(
    sugar_key: str,
    receptor_aromatics: List[Dict[str, str]],
) -> List[CHPiContact]:
    """
    Estimate CH-pi contacts from sugar face profile and receptor aromatics.

    For cases where crystal structure contacts aren't manually extracted.
    Uses the sugar's best stacking face and assumes ideal geometry.

    Parameters
    ----------
    sugar_key : str
        Key into CH_PI_PROFILES.
    receptor_aromatics : list of dict
        Each dict has 'residue' (e.g. 'Trp181') and 'type' (e.g. 'Trp').
        Optionally 'face' ('alpha' or 'beta') to specify which face.

    Returns
    -------
    List of CHPiContact (one per axial CH per aromatic that stacks that face).
    """
    profile = get_ch_pi_profile(sugar_key)
    if profile is None:
        return []

    contacts = []
    for arom in receptor_aromatics:
        face_pref = arom.get('face', None)
        if face_pref == 'beta' and profile.beta_face.stackable:
            face = profile.beta_face
        elif face_pref == 'alpha' and profile.alpha_face.stackable:
            face = profile.alpha_face
        else:
            face = profile.best_face

        if not face.stackable:
            continue

        for ch_pos in face.axial_CH_positions:
            contacts.append(CHPiContact(
                sugar_position=ch_pos,
                aromatic_residue=arom['residue'],
                aromatic_type=arom['type'],
            ))

    return contacts


# =====================================================================
# SELF-TEST
# =====================================================================

def _self_test():
    # 1. Parameter in expected range
    assert -4.0 < EPS_CH_PI < -1.0

    # 2. Aromatic hierarchy
    assert AROMATIC_WEIGHT['Trp'] > AROMATIC_WEIGHT['Tyr']
    assert AROMATIC_WEIGHT['Tyr'] > AROMATIC_WEIGHT['Phe']
    assert AROMATIC_WEIGHT['Phe'] > AROMATIC_WEIGHT['His']

    # 3. Distance weighting peaks at optimal
    assert f_distance(D_OPTIMAL) > 0.99
    assert f_distance(D_CUTOFF + 0.1) == 0.0
    assert f_distance(D_OPTIMAL) > f_distance(4.0)

    # 4. Angle weighting peaks at perpendicular
    assert f_angle(0) > 0.99
    assert f_angle(90) == 0.0

    # 5. Empty contacts = zero
    r = score_ch_pi([])
    assert r['dG_ch_pi'] == 0.0

    # 6. Man alpha face has 3 CH
    p = CH_PI_PROFILES['aMan']
    assert p.alpha_face.n_axial_CH == 3
    assert p.beta_face.stackable is False

    # 7. bGlcNAc is sandwichable (WGA/hevein target)
    p = CH_PI_PROFILES['bGlcNAc']
    assert p.sandwich_possible is True


_self_test()
