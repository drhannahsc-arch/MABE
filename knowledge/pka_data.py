"""
knowledge/pka_data.py - pKa values for donor functional groups.

Every donor functional group has a protonation equilibrium:
    D⁻ + H⁺ ⇌ DH     Ka = [D⁻][H⁺] / [DH]

Below pKa: protonated (DH) — lone pair occupied, cannot coordinate metal.
Above pKa: deprotonated (D⁻) — lone pair available for donation.

Henderson-Hasselbalch:
    fraction_deprotonated = 1 / (1 + 10^(pKa - pH))

Metal-assisted deprotonation:
    Metal binding SHIFTS pKa downward because the metal stabilizes the
    deprotonated form. A thiol with pKa 8.5 that coordinates Hg²⁺ might
    have an effective pKa of 4-5 because Hg-S bond formation provides
    enough energy to pull the proton off at lower pH.
    
    Shift magnitude depends on:
    - Metal-donor bond strength (HSAB match)
    - Metal charge (higher charge = more pKa depression)
    - Donor type (soft donors show larger shifts with soft metals)

CRITICAL DESIGN PRINCIPLE:
    At pH 3.5, an EDTA-type chelator (pKa_carboxylate ~ 4.5, pKa_amine ~ 10)
    has most carboxylates protonated and ALL amines protonated.
    Effective denticity drops from 6 to maybe 1-2.
    But a dithiol (pKa_thiol ~ 8.5) with Hg²⁺ has effective pKa ~ 3-4
    due to metal-assisted deprotonation → still functional at pH 3.5.
    
    This is why soft donors work at low pH for soft metals.
    This is why EDTA fails for lead at pH < 4.
    The engine MUST know this.
"""

import math
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════
# pKa values for donor functional groups
# These are INTRINSIC pKa in water at 25°C, zero ionic strength
# Organized by donor atom type
# ═══════════════════════════════════════════════════════════════════════════

DONOR_PKA = {
    # Oxygen donors
    "carboxylate": {"pka": 4.5, "atom": "O", "protonated_form": "COOH", "deprotonated_form": "COO⁻",
        "notes": "Acetate-like. Range 3.5-5.5 depending on substituents."},
    "hydroxamate": {"pka": 9.0, "atom": "O", "protonated_form": "CONHOH", "deprotonated_form": "CONHO⁻",
        "notes": "Siderophore-type. Strong Fe3+ binding despite high pKa."},
    "phenolate": {"pka": 10.0, "atom": "O", "protonated_form": "PhOH", "deprotonated_form": "PhO⁻",
        "notes": "Catechol: pKa1=9.2, pKa2=13.0. Siderophore chemistry."},
    "phosphonate": {"pka": 6.5, "atom": "O", "protonated_form": "PO₃H₂", "deprotonated_form": "PO₃H⁻/PO₃²⁻",
        "notes": "pKa1~1.5 (always deprot), pKa2~6.5 (relevant). Two deprotonation steps."},
    "hydroxyl_alcohol": {"pka": 16.0, "atom": "O", "protonated_form": "ROH", "deprotonated_form": "RO⁻",
        "notes": "Alcohol. Almost never deprotonated in aqueous. Donates as neutral OH."},
    "water": {"pka": 15.7, "atom": "O", "protonated_form": "H₂O", "deprotonated_form": "OH⁻",
        "notes": "Reference. Coordinated water donates as neutral."},
    "silanol": {"pka": 7.0, "atom": "O", "protonated_form": "SiOH", "deprotonated_form": "SiO⁻",
        "notes": "Silica surface. pKa varies 6-8 with surface chemistry."},

    # Nitrogen donors
    "amine_primary": {"pka": 10.5, "atom": "N", "protonated_form": "RNH₃⁺", "deprotonated_form": "RNH₂",
        "notes": "Primary amine. Dead at pH < 8."},
    "amine_secondary": {"pka": 11.0, "atom": "N", "protonated_form": "R₂NH₂⁺", "deprotonated_form": "R₂NH",
        "notes": "Secondary amine. Dead at pH < 9."},
    "amine_tertiary": {"pka": 10.0, "atom": "N", "protonated_form": "R₃NH⁺", "deprotonated_form": "R₃N",
        "notes": "Tertiary amine. Dead at pH < 8."},
    "imidazole": {"pka": 6.5, "atom": "N", "protonated_form": "ImH⁺", "deprotonated_form": "Im",
        "notes": "Histidine sidechain. Active above pH ~5-6."},
    "pyridine": {"pka": 5.2, "atom": "N", "protonated_form": "PyH⁺", "deprotonated_form": "Py",
        "notes": "Aromatic N. Better at low pH than amines."},
    "amide": {"pka": -1.0, "atom": "N", "protonated_form": "RCONH₂", "deprotonated_form": "RCONH⁻",
        "notes": "Amide N. Donates as neutral (not deprotonation-dependent). pKa of conjugate acid ~ -1."},
    "imine_schiff": {"pka": 7.0, "atom": "N", "protonated_form": "RCH=NH⁺R", "deprotonated_form": "RCH=NR",
        "notes": "Schiff base. Moderate pKa."},

    # Sulfur donors
    "thiolate": {"pka": 8.5, "atom": "S", "protonated_form": "RSH", "deprotonated_form": "RS⁻",
        "notes": "Cysteine-like. pKa 8-10. Strong metal-assisted shift."},
    "thioether": {"pka": -2.0, "atom": "S", "protonated_form": "RSR (no deprot needed)", "deprotonated_form": "RSR",
        "notes": "Methionine-like. Donates as neutral. No protonation issue."},
    "dithiocarbamate": {"pka": 3.5, "atom": "S", "protonated_form": "R₂NCSSH", "deprotonated_form": "R₂NCSS⁻",
        "notes": "Low pKa. Excellent at acidic pH. Classic mining reagent."},
    "xanthate": {"pka": 1.5, "atom": "S", "protonated_form": "ROCSS-H", "deprotonated_form": "ROCSS⁻",
        "notes": "Very low pKa. Always active. Mineral flotation reagent."},
    "thiourea": {"pka": -1.0, "atom": "S", "protonated_form": "(NH₂)₂CS", "deprotonated_form": "(NH₂)₂CS",
        "notes": "Neutral donor. No deprotonation needed. Active at any pH."},

    # Phosphorus donors
    "phosphine": {"pka": -2.0, "atom": "P", "protonated_form": "R₃PH⁺", "deprotonated_form": "R₃P",
        "notes": "Neutral donor. pKa of conjugate acid very low. Always available."},
}


# ═══════════════════════════════════════════════════════════════════════════
# Metal-assisted pKa shift data
# How much a metal shifts the pKa of a donor downward
# Depends on HSAB match and metal charge
# ═══════════════════════════════════════════════════════════════════════════

# Shift magnitude (pKa units lowered) by donor_type × metal_hsab
METAL_ASSISTED_PKA_SHIFT = {
    # (donor_hsab, metal_hsab) → approximate pKa units lowered
    ("soft", "soft"): 4.0,       # thiol + Hg/Au: massive shift (pKa drops 4 units)
    ("soft", "borderline"): 2.5, # thiol + Pb/Cu: significant shift
    ("soft", "hard"): 1.0,       # thiol + Fe3+: modest shift
    ("hard", "hard"): 2.0,       # carboxylate + Fe3+/Al3+: moderate shift
    ("hard", "borderline"): 1.0, # carboxylate + Pb: modest
    ("hard", "soft"): 0.5,       # carboxylate + Au: minimal
    ("borderline", "borderline"): 2.0, # amine + Cu/Ni: moderate
    ("borderline", "soft"): 1.5, # amine + Au: some shift
    ("borderline", "hard"): 1.0, # amine + Fe3+: modest
}

# Additional shift from metal charge: higher charge = more depression
# Each unit of charge beyond +2 adds ~0.5 pKa units of shift
CHARGE_PKA_SHIFT = 0.5  # per charge unit above +2


def fraction_deprotonated(pka: float, ph: float) -> float:
    """Henderson-Hasselbalch: fraction of donor in active (deprotonated) form."""
    if ph - pka > 10:
        return 1.0
    if pka - ph > 10:
        return 0.0
    return 1.0 / (1.0 + 10.0 ** (pka - ph))


def effective_pka(intrinsic_pka: float,
                   donor_hsab: str,
                   metal_hsab: str,
                   metal_charge: float) -> float:
    """
    Compute effective pKa accounting for metal-assisted deprotonation.
    
    Metal coordination stabilizes the deprotonated form, shifting pKa downward.
    """
    key = (donor_hsab, metal_hsab)
    shift = METAL_ASSISTED_PKA_SHIFT.get(key, 1.0)
    
    # Charge correction
    charge_extra = max(0, abs(metal_charge) - 2.0) * CHARGE_PKA_SHIFT
    
    return intrinsic_pka - shift - charge_extra


def classify_donor_group(donor_atom: str, structure_context: str = "") -> str:
    """
    Classify a donor atom into a functional group based on atom type
    and structural context.
    
    Returns key into DONOR_PKA dictionary.
    """
    context = structure_context.lower() if structure_context else ""
    
    if donor_atom == "S":
        if "thioether" in context or "methionine" in context:
            return "thioether"
        if "dithiocarbamate" in context:
            return "dithiocarbamate"
        if "xanthate" in context:
            return "xanthate"
        if "thiourea" in context or "allylthiourea" in context:
            return "thiourea"
        return "thiolate"  # default S donor
    
    elif donor_atom == "N":
        if "imidazole" in context or "histidine" in context or "vinylimidazole" in context:
            return "imidazole"
        if "pyridine" in context or "vinylpyridine" in context:
            return "pyridine"
        if "amide" in context:
            return "amide"
        if "imine" in context or "schiff" in context:
            return "imine_schiff"
        if "secondary" in context:
            return "amine_secondary"
        if "tertiary" in context:
            return "amine_tertiary"
        return "amine_primary"  # default N donor
    
    elif donor_atom == "O":
        if "carboxyl" in context or "acetate" in context or "methacrylic" in context:
            return "carboxylate"
        if "hydroxamate" in context or "siderophore" in context:
            return "hydroxamate"
        if "phenol" in context or "catechol" in context:
            return "phenolate"
        if "phosphonate" in context:
            return "phosphonate"
        if "silanol" in context or "silica" in context or "APTES" in context:
            return "silanol"
        return "carboxylate"  # default O donor
    
    elif donor_atom == "P":
        return "phosphine"
    
    return "carboxylate"  # fallback


def get_donor_pka(donor_atom: str, structure_context: str = "") -> tuple[float, str]:
    """
    Get pKa for a donor atom given structural context.
    Returns (pKa, functional_group_name).
    """
    group = classify_donor_group(donor_atom, structure_context)
    entry = DONOR_PKA.get(group, {"pka": 7.0})
    return entry["pka"], group
