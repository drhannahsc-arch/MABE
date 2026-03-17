"""
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

# ── ITC-CALIBRATED TOTAL TdS (intra-well + inter-well) ──────────────
# Anchor: Dam & Brewer 2002, ConA triMan vs MeaMan: delta_TdS = 18.0 kJ/mol
# QM barrier ratios from Kirschner 2008 set the relative per-linkage values.
# Per-torsion: 3.3-3.7 kJ/mol (brackets Mammen 3.4 consensus).
ITC_CALIBRATED_TDS = {
    'alpha1-2': 6.58, 'alpha1-3': 6.88, 'alpha1-4': 6.94,
    'alpha1-6': 11.12,
    'beta1-2':  6.54, 'beta1-3':  6.54, 'beta1-4':  6.54,
    'beta1-6':  11.14,
    'alpha2-3': 5.0,  # Sialic acid: restricted by carboxylate (estimate)
}

K_BRANCH_PENALTY = 3.3  # kJ/mol per branch point (ITC-calibrated)


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
        if lt in ITC_CALIBRATED_TDS:
            tds = ITC_CALIBRATED_TDS[lt]
        elif lt in LINKAGE_REGISTRY:
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
    # ITC-calibrated checks
    assert ITC_CALIBRATED_TDS['alpha1-6'] > ITC_CALIBRATED_TDS['alpha1-3']
    assert ITC_CALIBRATED_TDS['beta1-4'] < ITC_CALIBRATED_TDS['alpha1-6']
    assert abs(ITC_CALIBRATED_TDS['alpha1-3'] + ITC_CALIBRATED_TDS['alpha1-6'] - 18.0) < 0.1
    # Trimannoside entropy check
    r2 = compute_conformational_entropy(['alpha1-3', 'alpha1-6'], n_branch_points=1)
    assert r2['TdS_total'] > 20.0  # ~21.3 kJ/mol

_self_test()
