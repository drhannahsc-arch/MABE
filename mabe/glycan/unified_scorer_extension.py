"""
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
