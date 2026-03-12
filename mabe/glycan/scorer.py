"""
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
