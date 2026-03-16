"""
knowledge/entropy_of_mixing.py — Translational + rotational entropy loss upon binding

When protein and ligand associate, the system loses 6 relative degrees
of freedom (3 translational + 3 rotational). This imposes a universal
entropic cost on ALL binding events regardless of chemistry.

The magnitude is well-established from multiple independent approaches:
  Zhou & Gilson 2009, Chem Rev 109:4092: comprehensive review, 25-30 kJ/mol
  Deng & Roux 2009, J Phys Chem B 113:2234: alchemical FEP, 24-32 kJ/mol
  Finkelstein & Janin 1989, Protein Eng 3:1: analytical, ~30 kJ/mol
  Hermans & Wang 1997, JACS 119:2707: FEP, 25-30 kJ/mol

The value has weak MW dependence (~25 for fragments, ~32 for peptides).
Vibrational recovery (6 new soft modes in the complex) is contentious:
  some authors include it (~18 kJ/mol recovery), others don't.
  We provide both gross and net values.

This term is currently ABSORBED into per-scaffold DG0 offsets.
Making it explicit enables cross-scaffold prediction without anchoring.

NOTE: this module does NOT attempt Sackur-Tetrode from first principles
(which gives gas-phase absolute entropy, not solution binding loss).
Instead it uses a consensus-calibrated formula matching the literature range.
"""

import math


# ═══════════════════════════════════════════════════════════════════════════
# CONSENSUS-CALIBRATED FORMULA
# ═══════════════════════════════════════════════════════════════════════════

# -TDeltaS_mix(gross) = A + B * ln(MW)
# Calibrated to match:
#   MW=150 -> 25 kJ/mol (small fragment)
#   MW=500 -> 30 kJ/mol (typical drug)
#   MW=800 -> 32 kJ/mol (macrocycle/peptide)
#
# Sources: Zhou & Gilson 2009 consensus, Deng & Roux 2009 FEP values

_A = 4.1     # intercept (kJ/mol)
_B = 4.18    # slope per ln(MW) (kJ/mol)

# Vibrational recovery: 6 new soft intermolecular modes in the complex
# Each mode at ~50 cm-1 recovers ~3 kJ/mol entropy at 298 K
# Source: Gilson et al. 1997, Biophys J 72:1047
VIB_RECOVERY_PER_MODE = 3.0   # kJ/mol per new vibrational mode
N_NEW_MODES = 6               # 3 trans + 3 rot -> 6 vibrations
VIB_RECOVERY_TOTAL = VIB_RECOVERY_PER_MODE * N_NEW_MODES  # 18 kJ/mol


def total_mixing_entropy(mw_Da=350.0):
    """Total entropy cost of bimolecular association.

    Args:
        mw_Da: ligand molecular weight in daltons

    Returns:
        dict with:
            gross_kJ: trans+rot loss BEFORE vibrational recovery (positive)
            vib_recovery_kJ: entropy recovered by new vibrational modes
            net_kJ: gross - recovery (positive = unfavorable)
    """
    mw = max(mw_Da, 50.0)  # clamp to avoid log of tiny values
    gross = _A + _B * math.log(mw)

    return {
        "gross_kJ": gross,
        "vib_recovery_kJ": VIB_RECOVERY_TOTAL,
        "net_kJ": gross - VIB_RECOVERY_TOTAL,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CONSENSUS VALUES (for quick estimation without MW)
# ═══════════════════════════════════════════════════════════════════════════

CONSENSUS_MIXING_ENTROPY = {
    "fragment":         25.0,   # MW ~100-200
    "small_molecule":   27.0,   # MW ~200-300
    "drug_like":        28.5,   # MW ~300-500 (median drug)
    "large_molecule":   31.0,   # MW ~500-800
    "peptide":          33.0,   # MW ~800-2000
    "default":          28.0,   # MW ~350
}


def mixing_entropy_quick(mw_Da=None, category=None, include_vib_recovery=False):
    """Quick estimate of -TDeltaS_mix in kJ/mol.

    Args:
        mw_Da: molecular weight (uses calibrated formula if provided)
        category: "fragment", "drug_like", "large_molecule", etc.
        include_vib_recovery: if True, subtract vibrational recovery

    Returns: -TDeltaS_mix in kJ/mol (positive = unfavorable)
    """
    if mw_Da is not None:
        result = total_mixing_entropy(mw_Da)
        return result["net_kJ"] if include_vib_recovery else result["gross_kJ"]

    gross = CONSENSUS_MIXING_ENTROPY.get(
        category, CONSENSUS_MIXING_ENTROPY["default"])

    if include_vib_recovery:
        return gross - VIB_RECOVERY_TOTAL
    return gross