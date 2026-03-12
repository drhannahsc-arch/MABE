"""
mabe/glycan/descriptors.py — Glycan binding descriptor constructor
===================================================================

Constructs a UniversalComplex-compatible dict from glycan binding data.
This is the glycan equivalent of from_metal_ligand() and from_host_guest().
"""

from typing import Optional, Dict, Any
from mabe.glycan.sugar_properties import SugarPropertyCard, get_sugar_card
from mabe.glycan.contact_map import GlycanContactMap


def from_glycan_binding(
    sugar_key: str,
    contacts: GlycanContactMap,
    receptor_name: str = '',
    log_Ka: Optional[float] = None,
    dG_exp_kj: Optional[float] = None,
    dH_exp_kj: Optional[float] = None,
    temperature_C: float = 25.0,
    pH: float = 7.0,
    source: str = '',
    beta_context: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Construct a glycan binding entry compatible with UniversalComplex.
    
    Returns a dict that can be used to populate UniversalComplex fields.
    The unified scorer reads these fields to activate glycan terms.
    
    Parameters
    ----------
    sugar_key : str
        Key into SUGAR_LIBRARY (e.g. 'aMan', 'aGlc', 'aGal')
    contacts : GlycanContactMap
        Contact map from crystal structure
    receptor_name : str
        Name of the receptor/lectin
    log_Ka : float or None
        Experimental binding constant (log scale)
    dG_exp_kj : float or None
        Experimental free energy (kJ/mol)
    dH_exp_kj : float or None
        Experimental enthalpy (kJ/mol, from ITC)
    """
    sugar = get_sugar_card(sugar_key)

    return {
        # Identity
        'name': f"{receptor_name}:{sugar.three_letter}",
        'host_name': receptor_name,
        'guest_name': sugar.name,
        'binding_mode': 'glycan_recognition',

        # Thermodynamics
        'log_K': log_Ka,
        'dG_kj': dG_exp_kj,
        'dH_kj': dH_exp_kj,
        'temperature_C': temperature_C,
        'pH': pH,

        # Glycan-specific fields (NEW — these activate glycan terms)
        'sugar_property_card': sugar,
        'glycan_contact_map': contacts,
        'beta_context': beta_context,

        # Standard fields for unified scorer compatibility
        'guest_n_hb_donors': len(sugar.hydroxyls),
        'guest_n_hb_acceptors': len(sugar.hydroxyls),
        'n_hbonds_formed': sum(contacts.n_hbonds_per_oh),
        'n_pi_contacts': contacts.total_ch_pi,
        'binding_mode': 'glycan_recognition',
        'source': source,
    }
