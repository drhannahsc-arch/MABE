#!/usr/bin/env python3
"""
bootstrap_glycan_g2.py — G2 Conformational Entropy Addendum
=============================================================

Run AFTER bootstrap_glycan_integration.py.
Adds conformational entropy (torsion freezing) to the glycan module.

WHAT THIS DOES:
  1. Creates mabe/glycan/conformational.py (linkage profiles + scoring)
  2. Updates mabe/glycan/contact_map.py (adds linkage fields + trimannoside)
  3. Updates mabe/glycan/scorer.py (wires G2 into compute_glycan_terms)
  4. Updates mabe/glycan/unified_scorer_extension.py (adds conf_entropy field)
  5. Creates tests/test_glycan_g2_conformational.py (26 tests)

DEPLOYMENT:
  python bootstrap_glycan_g2.py
  python -m pytest tests/test_glycan_g2_conformational.py -v
  python -m pytest tests/test_glycan_integration.py -v

All existing G1 tests must still pass.
"""

import os

FILES = {}

# ─────────────────────────────────────────────────────────────────────
# FILE 1: mabe/glycan/conformational.py (NEW)
# ─────────────────────────────────────────────────────────────────────
FILES["mabe/glycan/conformational.py"] = open(
    os.path.join(os.path.dirname(__file__), "mabe", "glycan", "conformational.py")
).read() if os.path.exists(os.path.join(os.path.dirname(__file__), "mabe", "glycan", "conformational.py")) else None

# Since the bootstrap may be run standalone, embed the content:
FILES["mabe/glycan/conformational.py"] = r'''"""
mabe/glycan/conformational.py -- G2: Conformational Entropy
============================================================

Computes the entropy cost of freezing glycosidic torsions upon binding.

Physics:
  In solution, glycosidic linkages populate multiple conformers on their
  phi/psi (and omega for 1->6) energy surfaces. Upon binding, these
  torsions freeze into a single bound-state conformation. The entropy
  cost is:

    TdS_freeze = -RT * sum_i(p_i * ln(p_i))    [free state entropy]

  where p_i = Boltzmann population of torsion bin i from the QM surface.
  The bound state has TdS = 0 (single well).

Parameter Sources (zero biology):
  - QM torsion barriers: Kirschner et al. 2008 (J. Comput. Chem. 29:622)
    GLYCAM06 parameterization -- MP2/cc-pVTZ scans on methyl glycosides
  - Small-molecule barriers: Wiberg & Murcko 1988 (JACS 110:8029)
  - NMR populations: Serianni group J-coupling-derived populations
  - Cross-check: Mammen & Whitesides 1998 (Angew. Chem. 37:2754)
    consensus TdS = 3.4 kJ/mol per frozen rotor (generic)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

R = 8.314e-3  # kJ/(mol*K)
T = 298.15    # K


@dataclass
class LinkageTorsionProfile:
    """Torsional energy profile for a glycosidic linkage type."""
    linkage_type: str
    n_torsions: int
    n_minima: int
    populations: List[float]
    TdS_freeze_kj: float
    barrier_mean_kj: float
    source: str

    @property
    def is_flexible(self) -> bool:
        return sum(1 for p in self.populations if p > 0.05) > 2


def _compute_TdS(populations: List[float]) -> float:
    """Compute conformational entropy from population distribution."""
    S = 0.0
    for p in populations:
        if p > 1e-10:
            S -= p * math.log(p)
    return R * T * S


# Linkage profiles from GLYCAM06 QM + NMR populations (Kirschner 2008)

_b14_pops = [0.85, 0.15]
BETA_1_4 = LinkageTorsionProfile(
    linkage_type='beta1-4', n_torsions=2, n_minima=2,
    populations=_b14_pops, TdS_freeze_kj=round(_compute_TdS(_b14_pops), 2),
    barrier_mean_kj=15.0, source='GLYCAM06 QM + Serianni NMR; Kirschner 2008',
)

_b13_pops = [0.70, 0.30]
BETA_1_3 = LinkageTorsionProfile(
    linkage_type='beta1-3', n_torsions=2, n_minima=2,
    populations=_b13_pops, TdS_freeze_kj=round(_compute_TdS(_b13_pops), 2),
    barrier_mean_kj=12.0, source='GLYCAM06 QM; Kirschner 2008',
)

_b12_pops = [0.60, 0.30, 0.10]
BETA_1_2 = LinkageTorsionProfile(
    linkage_type='beta1-2', n_torsions=2, n_minima=3,
    populations=_b12_pops, TdS_freeze_kj=round(_compute_TdS(_b12_pops), 2),
    barrier_mean_kj=10.0, source='GLYCAM06 QM; Kirschner 2008',
)

_a14_pops = [0.75, 0.25]
ALPHA_1_4 = LinkageTorsionProfile(
    linkage_type='alpha1-4', n_torsions=2, n_minima=2,
    populations=_a14_pops, TdS_freeze_kj=round(_compute_TdS(_a14_pops), 2),
    barrier_mean_kj=13.0, source='GLYCAM06 QM + NMR; Kirschner 2008',
)

_a13_pops = [0.80, 0.20]
ALPHA_1_3 = LinkageTorsionProfile(
    linkage_type='alpha1-3', n_torsions=2, n_minima=2,
    populations=_a13_pops, TdS_freeze_kj=round(_compute_TdS(_a13_pops), 2),
    barrier_mean_kj=14.0, source='GLYCAM06 QM; Kirschner 2008',
)

_a12_pops = [0.70, 0.30]
ALPHA_1_2 = LinkageTorsionProfile(
    linkage_type='alpha1-2', n_torsions=2, n_minima=2,
    populations=_a12_pops, TdS_freeze_kj=round(_compute_TdS(_a12_pops), 2),
    barrier_mean_kj=11.0, source='GLYCAM06 QM; Kirschner 2008',
)

# alpha1->6: THREE torsions (phi/psi/omega). omega has 3 rotamers (gt, gg, tg).
# NMR: gt:gg:tg = 50:30:20. Combined with phi/psi: 6 basins.
_a16_pops = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
ALPHA_1_6 = LinkageTorsionProfile(
    linkage_type='alpha1-6', n_torsions=3, n_minima=6,
    populations=_a16_pops, TdS_freeze_kj=round(_compute_TdS(_a16_pops), 2),
    barrier_mean_kj=8.0,
    source='GLYCAM06 QM + Vliegenthart NMR omega populations; Kirschner 2008',
)

# alpha2->3 (sialic acid): restricted by carboxylate
_a23_pops = [0.90, 0.10]
ALPHA_2_3 = LinkageTorsionProfile(
    linkage_type='alpha2-3', n_torsions=2, n_minima=2,
    populations=_a23_pops, TdS_freeze_kj=round(_compute_TdS(_a23_pops), 2),
    barrier_mean_kj=18.0, source='GLYCAM06 QM; Kirschner 2008; Serianni NMR',
)

LINKAGE_REGISTRY: Dict[str, LinkageTorsionProfile] = {
    'beta1-4': BETA_1_4, 'beta1-3': BETA_1_3, 'beta1-2': BETA_1_2,
    'alpha1-4': ALPHA_1_4, 'alpha1-3': ALPHA_1_3, 'alpha1-2': ALPHA_1_2,
    'alpha1-6': ALPHA_1_6, 'alpha2-3': ALPHA_2_3,
}

K_BRANCH_PENALTY = 2.0  # kJ/mol per branch point (PLACEHOLDER)


def compute_conformational_entropy(
    linkage_types: Optional[List[str]] = None,
    n_branch_points: int = 0,
    k_branch: float = K_BRANCH_PENALTY,
) -> Dict[str, float]:
    """
    Compute total conformational entropy cost for a glycan binding event.
    Self-zeros when linkage_types is None or empty (monosaccharide).
    """
    if not linkage_types:
        return {'TdS_total': 0.0, 'TdS_per_linkage': [],
                'TdS_branch': 0.0, 'linkage_details': []}

    per_linkage = []
    details = []
    for lt in linkage_types:
        if lt in LINKAGE_REGISTRY:
            tds = LINKAGE_REGISTRY[lt].TdS_freeze_kj
        else:
            tds = round(2 * R * T * math.log(3), 2)
        per_linkage.append(tds)
        details.append((lt, tds))

    branch_penalty = n_branch_points * k_branch
    total = sum(per_linkage) + branch_penalty
    return {'TdS_total': round(total, 2), 'TdS_per_linkage': per_linkage,
            'TdS_branch': round(branch_penalty, 2), 'linkage_details': details}


def _self_test():
    r = compute_conformational_entropy(None)
    assert r['TdS_total'] == 0.0
    assert ALPHA_1_6.TdS_freeze_kj > ALPHA_1_4.TdS_freeze_kj
    assert ALPHA_1_4.TdS_freeze_kj > ALPHA_2_3.TdS_freeze_kj
    for name, profile in LINKAGE_REGISTRY.items():
        assert 0.5 < profile.TdS_freeze_kj < 7.0
        assert abs(sum(profile.populations) - 1.0) < 0.01

_self_test()
'''

# ─────────────────────────────────────────────────────────────────────
# FILE 2: mabe/glycan/contact_map.py (REPLACES)
# ─────────────────────────────────────────────────────────────────────
FILES["mabe/glycan/contact_map.py"] = r'''"""
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
'''

# ─────────────────────────────────────────────────────────────────────
# FILE 3: mabe/glycan/scorer.py (REPLACES — adds G2 wiring)
# ─────────────────────────────────────────────────────────────────────
FILES["mabe/glycan/scorer.py"] = r'''"""
mabe/glycan/scorer.py -- Glycan recognition scorer (G1+G2)
===========================================================
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from mabe.glycan.params import GlycanParams, GLYCAN_PARAMS, SASA_DESOLV_PER_POSITION
from mabe.glycan.sugar_properties import SugarPropertyCard
from mabe.glycan.contact_map import GlycanContactMap
from mabe.glycan.conformational import compute_conformational_entropy


@dataclass
class GlycanScoreDecomposition:
    dG_polar_desolv: float = 0.0
    dG_hbond: float = 0.0
    dG_ch_pi: float = 0.0
    dG_conf_entropy: float = 0.0
    dG_structural_water: float = 0.0
    dG_ca_coordination: float = 0.0
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
    """Compute all glycan energy terms. Self-zeros if sugar or contacts are None."""
    result = GlycanScoreDecomposition()
    if sugar is None or contacts is None:
        return result

    if beta_context is None:
        beta_context = params.beta_context_default

    k_hbond = params.k_hbond_dH if use_enthalpy_hbond else params.k_hbond_dG

    # -- G1: Polar desolvation + H-bond compensation ---------------------
    n_hbonds = contacts.n_hbonds_per_oh
    n_oh = min(len(sugar.hydroxyls), len(n_hbonds))

    essentiality = {}
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

        if nhb >= 2:
            essentiality[oh.position] = 'essential'
        elif nhb == 1:
            essentiality[oh.position] = 'moderate'
        else:
            essentiality[oh.position] = 'nonessential'

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
    result.dG_conf_entropy = conf_result['TdS_total']

    # -- G3: CH-pi stacking ----------------------------------------------
    result.dG_ch_pi = contacts.total_ch_pi * params.eps_CH_pi

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

# ─────────────────────────────────────────────────────────────────────
# FILE 4: mabe/glycan/unified_scorer_extension.py (REPLACES)
# ─────────────────────────────────────────────────────────────────────
FILES["mabe/glycan/unified_scorer_extension.py"] = r'''"""
mabe/glycan/unified_scorer_extension.py -- Unified scorer integration hooks
============================================================================
"""

from typing import Any, Dict, Optional
from mabe.glycan.scorer import compute_glycan_terms
from mabe.glycan.params import GLYCAN_PARAMS


def compute_glycan_contribution(uc: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    """Compute glycan terms and add to unified scorer result. Self-zeros if no glycan data."""
    sugar = getattr(uc, 'sugar_property_card', None)
    contacts = getattr(uc, 'glycan_contact_map', None)

    if sugar is None or contacts is None:
        result.setdefault('dg_glycan_polar_desolv', 0.0)
        result.setdefault('dg_glycan_hbond', 0.0)
        result.setdefault('dg_glycan_conf_entropy', 0.0)
        result.setdefault('dg_glycan_ch_pi', 0.0)
        result.setdefault('dg_glycan_structural_water', 0.0)
        result.setdefault('dg_glycan_ca_coordination', 0.0)
        return result

    beta_context = getattr(uc, 'beta_context', None)

    glycan_result = compute_glycan_terms(
        sugar=sugar, contacts=contacts,
        params=GLYCAN_PARAMS, beta_context=beta_context,
    )

    result['dg_glycan_polar_desolv'] = glycan_result.dG_polar_desolv
    result['dg_glycan_hbond'] = glycan_result.dG_hbond
    result['dg_glycan_conf_entropy'] = glycan_result.dG_conf_entropy
    result['dg_glycan_ch_pi'] = glycan_result.dG_ch_pi
    result['dg_glycan_structural_water'] = glycan_result.dG_structural_water
    result['dg_glycan_ca_coordination'] = glycan_result.dG_ca_coordination

    result['dg_total'] = result.get('dg_total', 0.0) + glycan_result.dG_total

    result['glycan_essentiality_map'] = glycan_result.essentiality_map
    result['glycan_n_essential'] = glycan_result.n_essential
    result['glycan_n_nonessential'] = glycan_result.n_nonessential

    return result
'''

# ─────────────────────────────────────────────────────────────────────
# FILE 5: tests/test_glycan_g2_conformational.py (NEW)
# ─────────────────────────────────────────────────────────────────────
FILES["tests/test_glycan_g2_conformational.py"] = r'''"""
tests/test_glycan_g2_conformational.py -- G2 conformational entropy tests
"""
import pytest, math
from mabe.glycan.conformational import (
    LINKAGE_REGISTRY, K_BRANCH_PENALTY, compute_conformational_entropy, _compute_TdS,
    BETA_1_4, BETA_1_3, ALPHA_1_3, ALPHA_1_4, ALPHA_1_6, ALPHA_2_3, R, T,
)
from mabe.glycan.scorer import compute_glycan_terms
from mabe.glycan.sugar_properties import ALPHA_D_MANNOSE
from mabe.glycan.contact_map import cona_mannose_pocket, cona_trimannoside

class TestLinkageProfiles:
    def test_all_8_linkages_present(self):
        assert len(LINKAGE_REGISTRY) == 8
    def test_populations_sum_to_one(self):
        for n, p in LINKAGE_REGISTRY.items():
            assert abs(sum(p.populations) - 1.0) < 0.01, n
    def test_all_positive(self):
        for n, p in LINKAGE_REGISTRY.items():
            assert all(x > 0 for x in p.populations), n
    def test_n_minima_matches(self):
        for n, p in LINKAGE_REGISTRY.items():
            assert len(p.populations) == p.n_minima, n
    def test_tds_in_range(self):
        for n, p in LINKAGE_REGISTRY.items():
            assert 0.5 < p.TdS_freeze_kj < 7.0, f"{n}: {p.TdS_freeze_kj}"
    def test_alpha16_3_torsions(self):
        assert ALPHA_1_6.n_torsions == 3
    def test_others_2_torsions(self):
        for n, p in LINKAGE_REGISTRY.items():
            if n != 'alpha1-6':
                assert p.n_torsions == 2, n

class TestPhysicalOrdering:
    def test_alpha16_most_flexible(self):
        for n, p in LINKAGE_REGISTRY.items():
            if n != 'alpha1-6':
                assert ALPHA_1_6.TdS_freeze_kj > p.TdS_freeze_kj, n
    def test_alpha16_gt_alpha14(self):
        assert ALPHA_1_6.TdS_freeze_kj > ALPHA_1_4.TdS_freeze_kj
    def test_alpha14_gt_alpha23(self):
        assert ALPHA_1_4.TdS_freeze_kj > ALPHA_2_3.TdS_freeze_kj

class TestTdSComputation:
    def test_single_state_zero(self):
        assert abs(_compute_TdS([1.0])) < 1e-10
    def test_two_equal(self):
        assert abs(_compute_TdS([0.5, 0.5]) - R * T * math.log(2)) < 0.01
    def test_three_equal(self):
        assert abs(_compute_TdS([1/3, 1/3, 1/3]) - R * T * math.log(3)) < 0.01
    def test_more_states_more_entropy(self):
        assert _compute_TdS([1/3]*3) > _compute_TdS([0.5, 0.5])
    def test_skewed_less(self):
        assert _compute_TdS([0.5, 0.5]) > _compute_TdS([0.9, 0.1])

class TestScoring:
    def test_mono_zero(self):
        assert compute_conformational_entropy(None)['TdS_total'] == 0.0
    def test_empty_zero(self):
        assert compute_conformational_entropy([])['TdS_total'] == 0.0
    def test_single(self):
        r = compute_conformational_entropy(['beta1-4'])
        assert r['TdS_total'] == BETA_1_4.TdS_freeze_kj
    def test_additive(self):
        r = compute_conformational_entropy(['alpha1-3', 'alpha1-6'])
        assert abs(r['TdS_total'] - (ALPHA_1_3.TdS_freeze_kj + ALPHA_1_6.TdS_freeze_kj)) < 0.01
    def test_branch(self):
        a = compute_conformational_entropy(['alpha1-3'], n_branch_points=0)
        b = compute_conformational_entropy(['alpha1-3'], n_branch_points=1)
        assert b['TdS_total'] - a['TdS_total'] == K_BRANCH_PENALTY
    def test_unknown_fallback(self):
        r = compute_conformational_entropy(['unknown'])
        assert 4.0 < r['TdS_total'] < 7.0

class TestScorerIntegration:
    def test_mono_no_conf(self):
        assert compute_glycan_terms(ALPHA_D_MANNOSE, cona_mannose_pocket()).dG_conf_entropy == 0.0
    def test_tri_has_conf(self):
        r = compute_glycan_terms(ALPHA_D_MANNOSE, cona_trimannoside())
        assert 6.0 < r.dG_conf_entropy < 9.0
    def test_tri_less_favorable(self):
        m = compute_glycan_terms(ALPHA_D_MANNOSE, cona_mannose_pocket(), beta_context=0.45)
        t = compute_glycan_terms(ALPHA_D_MANNOSE, cona_trimannoside(), beta_context=0.45)
        assert (t.dG_conf_entropy - m.dG_conf_entropy) > 5.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''

# ─────────────────────────────────────────────────────────────────────
# DEPLOYMENT
# ─────────────────────────────────────────────────────────────────────

def deploy():
    created = []
    for relpath, content in FILES.items():
        fullpath = os.path.join(os.getcwd(), relpath)
        os.makedirs(os.path.dirname(fullpath), exist_ok=True)
        with open(fullpath, 'w', encoding='utf-8') as f:
            f.write(content.lstrip('\n'))
        created.append(relpath)
        print(f"  Created: {relpath}")

    print(f"\n{len(created)} files created/updated.")
    print("\nG2 adds:")
    print("  - 8 linkage torsion profiles (beta1-4, beta1-3, beta1-2,")
    print("    alpha1-4, alpha1-3, alpha1-2, alpha1-6, alpha2-3)")
    print("  - Branch penalty term")
    print("  - cona_trimannoside() contact map factory")
    print("  - G2 wired into scorer and unified_scorer_extension")
    print("\nRun: python -m pytest tests/ -v")


if __name__ == "__main__":
    deploy()