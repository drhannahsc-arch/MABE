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
