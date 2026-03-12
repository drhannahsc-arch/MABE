"""
mabe/glycan/unified_scorer_extension.py
========================================

Integration hooks for wiring glycan terms into unified_scorer_v2.

The unified scorer calls _compute_glycan() which self-zeros when
glycan fields are absent. This file provides the function signature
and the field-checking logic.

INTEGRATION INSTRUCTIONS:
  In unified_scorer_v2.py, add after existing _compute_* calls:

    from mabe.glycan.unified_scorer_extension import compute_glycan_contribution
    
    # In predict():
    result = compute_glycan_contribution(uc, result)

  That's it. The function checks for glycan fields and self-zeros
  if they're absent. No routing wall needed.
"""

from typing import Any, Dict, Optional
from mabe.glycan.scorer import compute_glycan_terms
from mabe.glycan.params import GLYCAN_PARAMS


def compute_glycan_contribution(uc: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute glycan energy terms and add to unified scorer result.
    
    Self-zeros if uc has no glycan data.
    
    Parameters
    ----------
    uc : UniversalComplex (or any object with attribute access)
        Must have optional fields:
          - sugar_property_card: SugarPropertyCard or None
          - glycan_contact_map: GlycanContactMap or None
          - beta_context: float or None
    result : dict
        Existing scorer result dict to extend with glycan terms.
    
    Returns
    -------
    result : dict with glycan terms added (or unchanged if no glycan data)
    """
    # Self-zero gate: check for glycan-specific fields
    sugar = getattr(uc, 'sugar_property_card', None)
    contacts = getattr(uc, 'glycan_contact_map', None)

    if sugar is None or contacts is None:
        # No glycan data → zero contribution
        result.setdefault('dg_glycan_polar_desolv', 0.0)
        result.setdefault('dg_glycan_hbond', 0.0)
        result.setdefault('dg_glycan_ch_pi', 0.0)
        result.setdefault('dg_glycan_structural_water', 0.0)
        result.setdefault('dg_glycan_ca_coordination', 0.0)
        return result

    beta_context = getattr(uc, 'beta_context', None)

    glycan_result = compute_glycan_terms(
        sugar=sugar,
        contacts=contacts,
        params=GLYCAN_PARAMS,
        beta_context=beta_context,
    )

    # Add glycan terms to result
    result['dg_glycan_polar_desolv'] = glycan_result.dG_polar_desolv
    result['dg_glycan_hbond'] = glycan_result.dG_hbond
    result['dg_glycan_ch_pi'] = glycan_result.dG_ch_pi
    result['dg_glycan_structural_water'] = glycan_result.dG_structural_water
    result['dg_glycan_ca_coordination'] = glycan_result.dG_ca_coordination

    # Add to total
    result['dg_total'] = result.get('dg_total', 0.0) + glycan_result.dG_total

    # Metadata
    result['glycan_essentiality_map'] = glycan_result.essentiality_map
    result['glycan_n_essential'] = glycan_result.n_essential
    result['glycan_n_nonessential'] = glycan_result.n_nonessential

    return result
