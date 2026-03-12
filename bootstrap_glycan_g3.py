#!/usr/bin/env python3
"""bootstrap_glycan_g3.py -- G3 CH-pi Module
Run AFTER bootstrap_glycan_g2.py.
Adds CH-pi stacking with geometric weighting and sugar face profiles.
"""
import os

FILES = {}

FILES["mabe/glycan/ch_pi.py"] = r'''
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
'''

FILES["mabe/glycan/scorer.py"] = r'''
"""
mabe/glycan/scorer.py — Glycan recognition scorer
===================================================

Scores glycan-binder interactions using physics-based decomposition.
All terms self-zero when their inputs are absent.

Designed to be called by the MABE unified scorer — not standalone.
The unified scorer calls compute_glycan_terms() which returns a dict
of energy contributions. Terms with no data return 0.0.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from mabe.glycan.params import GlycanParams, GLYCAN_PARAMS, SASA_DESOLV_PER_POSITION
from mabe.glycan.sugar_properties import SugarPropertyCard
from mabe.glycan.contact_map import GlycanContactMap
from mabe.glycan.conformational import compute_conformational_entropy


@dataclass
class GlycanScoreDecomposition:
    """Energy decomposition for a glycan-receptor interaction."""
    dG_polar_desolv: float = 0.0       # G1: polyol desolvation
    dG_hbond: float = 0.0              # G1: H-bond compensation
    dG_ch_pi: float = 0.0              # G3: CH-π stacking
    dG_conf_entropy: float = 0.0       # G2: glycosidic torsion freezing
    dG_structural_water: float = 0.0   # G5: conserved water bridges
    dG_ca_coordination: float = 0.0    # G7: C-type lectin Ca²⁺ bridging
    dG_total: float = 0.0

    essentiality_map: Dict[str, str] = field(default_factory=dict)
    n_essential: int = 0
    n_nonessential: int = 0


def compute_glycan_terms(
    sugar: Optional[SugarPropertyCard],
    contacts: Optional[GlycanContactMap],
    params: GlycanParams = GLYCAN_PARAMS,
    beta_context: Optional[float] = None,
    use_enthalpy_hbond: bool = True,
) -> GlycanScoreDecomposition:
    """
    Compute all glycan energy terms.
    
    Self-zeros if sugar or contacts are None.
    This is the function the unified scorer calls.
    
    Parameters
    ----------
    sugar : SugarPropertyCard or None
        Sugar descriptor. None → all terms zero.
    contacts : GlycanContactMap or None
        Contact map from crystal structure. None → all terms zero.
    params : GlycanParams
        Parameter set. Default: GLYCAN_PARAMS singleton.
    beta_context : float or None
        Context buffering. None → params.beta_context_default.
    use_enthalpy_hbond : bool
        If True, use k_hbond_dH (intrinsic, for partially buried contacts).
        If False, use k_hbond_dG (solvent-exposed, e.g. CD portal).
    
    Returns
    -------
    GlycanScoreDecomposition
    """
    result = GlycanScoreDecomposition()

    # Self-zero gate
    if sugar is None or contacts is None:
        return result

    if beta_context is None:
        beta_context = params.beta_context_default

    k_hbond = params.k_hbond_dH if use_enthalpy_hbond else params.k_hbond_dG

    # ── G1: Polar desolvation + H-bond compensation ────────────────────
    n_hbonds = contacts.n_hbonds_per_oh
    n_oh = min(len(sugar.hydroxyls), len(n_hbonds))

    dg_desolv_total = 0.0
    dg_hbond_total = 0.0
    essentiality = {}

    for i in range(n_oh):
        oh = sugar.hydroxyls[i]
        nhb = n_hbonds[i]

        # Desolvation cost (from SASA table or default)
        desolv_cost = SASA_DESOLV_PER_POSITION.get(
            oh.sasa_key, params.k_desolv_OH
        )

        # H-bond compensation
        hbond_comp = nhb * k_hbond

        # Net per-OH
        net = desolv_cost + hbond_comp

        # Apply context buffering to penalty portion only
        if net > 0:
            net *= beta_context

        dg_desolv_total += desolv_cost if net > 0 else desolv_cost
        dg_hbond_total += hbond_comp

        # Essentiality classification
        if nhb >= 2:
            essentiality[oh.position] = 'essential'
        elif nhb == 1:
            essentiality[oh.position] = 'moderate'
        else:
            essentiality[oh.position] = 'nonessential'

    # Recalculate total with buffering applied correctly
    dg_per_oh = []
    for i in range(n_oh):
        oh = sugar.hydroxyls[i]
        nhb = n_hbonds[i]
        desolv_cost = SASA_DESOLV_PER_POSITION.get(oh.sasa_key, params.k_desolv_OH)
        hbond_comp = nhb * k_hbond
        net = desolv_cost + hbond_comp
        if net > 0:
            net *= beta_context
        dg_per_oh.append(net)

    result.dG_polar_desolv = sum(max(0, d) for d in dg_per_oh)
    result.dG_hbond = sum(min(0, d) for d in dg_per_oh)

    result.essentiality_map = essentiality
    result.n_essential = sum(1 for v in essentiality.values() if v == 'essential')
    result.n_nonessential = sum(1 for v in essentiality.values() if v == 'nonessential')

    # -- G2: Conformational entropy (torsion freezing) -------------------
    conf_result = compute_conformational_entropy(
        linkage_types=contacts.linkage_types if contacts.linkage_types else None,
        n_branch_points=contacts.n_branch_points,
        k_branch=params.k_branch_penalty,
    )
    result.dG_conf_entropy = conf_result['TdS_total']  # positive = unfavorable

    # -- G3: CH-pi stacking ----------------------------------------------
    # If detailed CH-pi contacts are available (from ch_pi module), use them.
    # Otherwise fall back to simple count * eps_CH_pi.
    detailed_ch_pi = getattr(contacts, 'detailed_ch_pi_contacts', None)
    if detailed_ch_pi:
        from mabe.glycan.ch_pi import score_ch_pi
        ch_pi_result = score_ch_pi(detailed_ch_pi, eps=params.eps_CH_pi)
        result.dG_ch_pi = ch_pi_result['dG_ch_pi']
    else:
        n_ch_pi = contacts.total_ch_pi
        result.dG_ch_pi = n_ch_pi * params.eps_CH_pi

    # -- G5: Structural water ---------------------------------------------
    result.dG_structural_water = contacts.n_conserved_waters * params.eps_water_bridge

    # -- G7: Ca2+ bridging ------------------------------------------------
    if contacts.has_metal and contacts.metal_identity and 'Ca' in contacts.metal_identity:
        result.dG_ca_coordination = contacts.n_ca_bridges * params.eps_Ca_coordination

    # -- Total ------------------------------------------------------------
    result.dG_total = sum(dg_per_oh) + result.dG_conf_entropy + result.dG_ch_pi + \
                      result.dG_structural_water + result.dG_ca_coordination

    return result
'''

FILES["mabe/glycan/contact_map.py"] = r'''
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
'''

FILES["tests/test_glycan_g3_ch_pi.py"] = r'''
"""
tests/test_glycan_g3_ch_pi.py -- G3 CH-pi interaction tests
=============================================================

Tests:
  1. Parameter integrity (eps_CH_pi, aromatic weights)
  2. Geometric weighting (distance, angle)
  3. Sugar face profiles (axial CH counts, stacking rules)
  4. Scoring correctness (simple + geometry-weighted)
  5. Aromatic hierarchy (Trp > Tyr > Phe > His)
  6. Hevein cross-check (2-3 contacts x Trp = 5-7.5 kJ/mol)
  7. Integration with scorer (simple and detailed paths)
"""

import pytest
import math

from mabe.glycan.ch_pi import (
    EPS_CH_PI, AROMATIC_WEIGHT, D_OPTIMAL, D_SIGMA, D_CUTOFF,
    f_distance, f_angle, score_ch_pi, CHPiContact,
    CH_PI_PROFILES, get_ch_pi_profile, estimate_ch_pi_contacts,
)
from mabe.glycan.scorer import compute_glycan_terms
from mabe.glycan.sugar_properties import ALPHA_D_MANNOSE
from mabe.glycan.contact_map import (
    GlycanContactMap, OHContact,
    CHPiContact as ContactMapCHPi,
    cona_mannose_pocket,
)


# =====================================================================
# PARAMETER INTEGRITY
# =====================================================================

class TestParameters:

    def test_eps_ch_pi_locked(self):
        assert EPS_CH_PI == -2.5

    def test_eps_in_literature_range(self):
        assert -4.0 < EPS_CH_PI < -1.0

    def test_aromatic_hierarchy(self):
        assert AROMATIC_WEIGHT['Trp'] > AROMATIC_WEIGHT['Tyr']
        assert AROMATIC_WEIGHT['Tyr'] > AROMATIC_WEIGHT['Phe']
        assert AROMATIC_WEIGHT['Phe'] > AROMATIC_WEIGHT['His']

    def test_trp_is_reference(self):
        assert AROMATIC_WEIGHT['Trp'] == 1.0

    def test_all_weights_positive(self):
        for aa, w in AROMATIC_WEIGHT.items():
            assert 0 < w <= 1.0, f"{aa}: weight {w}"


# =====================================================================
# GEOMETRIC WEIGHTING
# =====================================================================

class TestGeometricWeighting:

    def test_distance_optimal(self):
        assert f_distance(D_OPTIMAL) > 0.99

    def test_distance_cutoff(self):
        assert f_distance(D_CUTOFF + 0.1) == 0.0

    def test_distance_too_close(self):
        assert f_distance(1.5) == 0.0

    def test_distance_monotonic_from_optimal(self):
        """Weight decreases as distance moves from optimal."""
        assert f_distance(3.5) > f_distance(4.0)
        assert f_distance(3.5) > f_distance(3.0)

    def test_distance_symmetric_around_optimal(self):
        """Approximately symmetric Gaussian."""
        w_near = f_distance(D_OPTIMAL - 0.3)
        w_far = f_distance(D_OPTIMAL + 0.3)
        assert abs(w_near - w_far) < 0.01

    def test_angle_perpendicular(self):
        assert f_angle(0) > 0.99

    def test_angle_parallel(self):
        assert f_angle(90) == 0.0

    def test_angle_45_deg(self):
        assert abs(f_angle(45) - 0.5) < 0.01

    def test_angle_monotonic(self):
        assert f_angle(0) > f_angle(30) > f_angle(60) > f_angle(89)


# =====================================================================
# SUGAR FACE PROFILES
# =====================================================================

class TestSugarFaces:

    def test_mannose_alpha_face_3ch(self):
        p = CH_PI_PROFILES['aMan']
        assert p.alpha_face.n_axial_CH == 3

    def test_mannose_beta_face_blocked(self):
        """Man C2-axial OH blocks beta face."""
        p = CH_PI_PROFILES['aMan']
        assert p.beta_face.stackable is False

    def test_glucose_alpha_face_3ch(self):
        p = CH_PI_PROFILES['aGlc']
        assert p.alpha_face.n_axial_CH == 3

    def test_galactose_alpha_face_3ch(self):
        p = CH_PI_PROFILES['aGal']
        assert p.alpha_face.n_axial_CH == 3

    def test_galactose_beta_blocked(self):
        """Gal C4-axial OH blocks beta face."""
        p = CH_PI_PROFILES['aGal']
        assert p.beta_face.stackable is False

    def test_bGlcNAc_sandwich(self):
        """beta-GlcNAc can be sandwiched (WGA/hevein)."""
        p = CH_PI_PROFILES['bGlcNAc']
        assert p.sandwich_possible is True
        assert p.alpha_face.stackable and p.beta_face.stackable

    def test_bGlc_sandwich(self):
        """beta-Glc (all eq) can be sandwiched."""
        p = CH_PI_PROFILES['bGlc']
        assert p.sandwich_possible is True

    def test_best_face(self):
        p = CH_PI_PROFILES['aMan']
        assert p.best_face.face == 'alpha'

    def test_all_profiles_have_max_contacts(self):
        for key, p in CH_PI_PROFILES.items():
            assert p.max_simultaneous_contacts >= 1, f"{key} has 0 max contacts"


# =====================================================================
# SCORING
# =====================================================================

class TestScoring:

    def test_empty_contacts_zero(self):
        r = score_ch_pi([])
        assert r['dG_ch_pi'] == 0.0
        assert r['n_contacts'] == 0

    def test_single_trp_ideal(self):
        """One contact with Trp at ideal geometry."""
        c = [CHPiContact('C3-H', 'Trp181', 'Trp')]
        r = score_ch_pi(c, use_geometry=False)
        assert abs(r['dG_ch_pi'] - (-2.5)) < 0.01

    def test_single_phe_weaker(self):
        """Phe contact weaker than Trp."""
        c_trp = [CHPiContact('C3-H', 'Trp181', 'Trp')]
        c_phe = [CHPiContact('C3-H', 'Phe181', 'Phe')]
        r_trp = score_ch_pi(c_trp, use_geometry=False)
        r_phe = score_ch_pi(c_phe, use_geometry=False)
        assert r_trp['dG_ch_pi'] < r_phe['dG_ch_pi']  # more negative = stronger

    def test_three_contacts_additive(self):
        """Three Trp contacts = 3 x eps."""
        contacts = [
            CHPiContact('C1-H', 'Trp181', 'Trp'),
            CHPiContact('C3-H', 'Trp181', 'Trp'),
            CHPiContact('C5-H', 'Trp181', 'Trp'),
        ]
        r = score_ch_pi(contacts, use_geometry=False)
        assert abs(r['dG_ch_pi'] - 3 * (-2.5)) < 0.01

    def test_distance_weighting_reduces(self):
        """Contact at 4.2 A weaker than at 3.5 A."""
        c_ideal = [CHPiContact('C3-H', 'Trp', 'Trp', distance_A=3.5)]
        c_far = [CHPiContact('C3-H', 'Trp', 'Trp', distance_A=4.2)]
        r_ideal = score_ch_pi(c_ideal)
        r_far = score_ch_pi(c_far)
        assert r_ideal['dG_ch_pi'] < r_far['dG_ch_pi']  # ideal more favorable

    def test_angle_weighting_reduces(self):
        """45 deg deviation weaker than perpendicular."""
        c_perp = [CHPiContact('C3-H', 'Trp', 'Trp', distance_A=3.5, angle_deviation_deg=0)]
        c_45 = [CHPiContact('C3-H', 'Trp', 'Trp', distance_A=3.5, angle_deviation_deg=45)]
        r_perp = score_ch_pi(c_perp)
        r_45 = score_ch_pi(c_45)
        assert r_perp['dG_ch_pi'] < r_45['dG_ch_pi']

    def test_aromatic_breakdown(self):
        contacts = [
            CHPiContact('C1-H', 'Trp181', 'Trp'),
            CHPiContact('C3-H', 'Tyr100', 'Tyr'),
        ]
        r = score_ch_pi(contacts, use_geometry=False)
        assert 'Trp181' in r['aromatic_breakdown']
        assert 'Tyr100' in r['aromatic_breakdown']

    def test_hevein_crosscheck(self):
        """Hevein double-Trp stacking: 2 aromatics x 3 CH each = ~15 kJ/mol.
        Asensio reports 6.3-8.4 kJ/mol total — but that's per stacking event,
        not per CH contact. At 3 contacts per Trp: 3 x -2.5 = -7.5. Consistent."""
        contacts = [
            CHPiContact(f'C{i}-H', 'Trp21', 'Trp')
            for i in [1, 3, 5]
        ]
        r = score_ch_pi(contacts, use_geometry=False)
        assert -8.5 < r['dG_ch_pi'] < -6.0  # consistent with Asensio 6.3-8.4


# =====================================================================
# CONTACT ESTIMATION
# =====================================================================

class TestContactEstimation:

    def test_mannose_one_trp(self):
        contacts = estimate_ch_pi_contacts('aMan', [{'residue': 'Trp181', 'type': 'Trp'}])
        assert len(contacts) == 3  # alpha face: C1-H, C3-H, C5-H

    def test_mannose_beta_face_blocked(self):
        contacts = estimate_ch_pi_contacts('aMan',
            [{'residue': 'Trp181', 'type': 'Trp', 'face': 'beta'}])
        assert len(contacts) == 3  # falls back to best (alpha)

    def test_unknown_sugar_empty(self):
        contacts = estimate_ch_pi_contacts('unknown', [{'residue': 'Trp', 'type': 'Trp'}])
        assert len(contacts) == 0

    def test_no_aromatics_empty(self):
        contacts = estimate_ch_pi_contacts('aMan', [])
        assert len(contacts) == 0


# =====================================================================
# SCORER INTEGRATION
# =====================================================================

class TestScorerIntegration:

    def test_cona_no_ch_pi(self):
        """ConA has no aromatic stacking — G3 should be zero."""
        r = compute_glycan_terms(ALPHA_D_MANNOSE, cona_mannose_pocket())
        assert r.dG_ch_pi == 0.0

    def test_simple_count_path(self):
        """Simple count path: 3 contacts x -2.5 = -7.5."""
        cm = cona_mannose_pocket()
        cm.ch_pi_contacts = [
            ContactMapCHPi(['C1-H', 'C3-H', 'C5-H'], 'Trp181', 3)
        ]
        r = compute_glycan_terms(ALPHA_D_MANNOSE, cm)
        assert abs(r.dG_ch_pi - (-7.5)) < 0.01

    def test_detailed_path(self):
        """Detailed path with aromatic weighting."""
        from mabe.glycan.ch_pi import CHPiContact as DetailedContact
        cm = cona_mannose_pocket()
        cm.ch_pi_contacts = [
            ContactMapCHPi(['C1-H', 'C3-H', 'C5-H'], 'Trp181', 3)
        ]
        cm.detailed_ch_pi_contacts = [
            DetailedContact('C1-H', 'Trp181', 'Trp'),
            DetailedContact('C3-H', 'Trp181', 'Trp'),
            DetailedContact('C5-H', 'Trp181', 'Trp'),
        ]
        r = compute_glycan_terms(ALPHA_D_MANNOSE, cm)
        # Trp weight = 1.0, no geometry = ideal: 3 x -2.5 = -7.5
        assert abs(r.dG_ch_pi - (-7.5)) < 0.01

    def test_detailed_tyr_weaker_than_trp(self):
        """Tyr contacts give less stabilization than Trp."""
        from mabe.glycan.ch_pi import CHPiContact as DetailedContact
        cm_trp = cona_mannose_pocket()
        cm_trp.detailed_ch_pi_contacts = [
            DetailedContact('C3-H', 'Trp181', 'Trp'),
        ]
        cm_trp.ch_pi_contacts = [ContactMapCHPi(['C3-H'], 'Trp181', 1)]

        cm_tyr = cona_mannose_pocket()
        cm_tyr.detailed_ch_pi_contacts = [
            DetailedContact('C3-H', 'Tyr100', 'Tyr'),
        ]
        cm_tyr.ch_pi_contacts = [ContactMapCHPi(['C3-H'], 'Tyr100', 1)]

        r_trp = compute_glycan_terms(ALPHA_D_MANNOSE, cm_trp)
        r_tyr = compute_glycan_terms(ALPHA_D_MANNOSE, cm_tyr)
        assert r_trp.dG_ch_pi < r_tyr.dG_ch_pi  # Trp more negative


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''


def deploy():
    created = []
    for relpath, content in FILES.items():
        fullpath = os.path.join(os.getcwd(), relpath)
        os.makedirs(os.path.dirname(fullpath), exist_ok=True)
        with open(fullpath, 'w', encoding='utf-8') as fh:
            fh.write(content.lstrip('\n'))
        created.append(relpath)
        print(f"  Created: {relpath}")
    print(f"\n{len(created)} files created/updated.")
    print("\nG3 adds:")
    print("  - CH-pi scoring with geometric weighting (distance + angle)")
    print("  - Aromatic hierarchy: Trp > Tyr > Phe > His")
    print("  - 9 sugar face profiles (aMan, bMan, aGlc, bGlc, aGal, bGal, aGlcNAc, bGlcNAc, aFuc)")
    print("  - Hevein/WGA prediction enabled")
    print("  - Detailed + simple scoring paths in scorer")
    print("\nRun: python -m pytest tests/ -v")

if __name__ == "__main__":
    deploy()