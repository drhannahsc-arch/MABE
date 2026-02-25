"""
realization_ranker/disqualification.py — Binary pass/fail gates.

These run BEFORE scoring. If a realization fails any gate, it's disqualified
with a reason string. No partial credit.

All gates are physics-grounded:
  - Scale mismatch: pocket size vs material positional precision
  - Donor unavailability: required donor atoms vs what the material can present
  - Condition incompatibility: deployment conditions vs material survival limits
"""

from dataclasses import dataclass
from typing import Optional
import math


# ═══════════════════════════════════════════════════════════════════════════
# Material capability constants — from physics, not opinion
# ═══════════════════════════════════════════════════════════════════════════

# Positional precision limits (nm) — minimum meaningful feature spacing
PRECISION_LIMITS = {
    "small_molecule":     0.1,    # Bond lengths, sub-angstrom control
    "chelator":           0.1,
    "porphyrin":          0.1,
    "crown_ether":        0.1,
    "peptide":            0.2,    # Side chain rotamer flexibility
    "protein":            0.3,    # Loop flexibility, B-factors
    "antibody_CDR":       0.3,
    "aptamer":            1.0,    # Nucleic acid fold uncertainty
    "dnazyme":            1.0,
    "DNA_origami":        1.5,    # Staple positioning ± 1-2 nm
    "MOF":                0.05,   # Crystallographic precision
    "crystal":            0.05,
    "ion_exchange_resin":  2.0,   # Amorphous, statistical cavities
}

# Maximum pocket/cavity size achievable (nm)
MAX_CAVITY = {
    "small_molecule":     0.8,
    "chelator":           0.6,
    "porphyrin":          0.5,    # Central cavity ~0.4 nm
    "crown_ether":        0.5,    # Cavity diameter 0.12-0.46 nm typical
    "peptide":            2.0,    # Cyclic peptides
    "protein":            5.0,    # Large binding clefts
    "antibody_CDR":       3.0,    # CDR loop pocket
    "aptamer":            3.0,    # Folded nucleic acid pocket
    "dnazyme":            2.0,
    "DNA_origami":        50.0,   # Interior of origami cage
    "MOF":                5.0,    # Pore diameters up to ~5 nm
    "crystal":            2.0,    # Lattice sites
    "ion_exchange_resin":  1.0,   # Functional group cavities
}

# Minimum pocket/cavity (nm) — below this the material can't form a defined pocket
MIN_CAVITY = {
    "small_molecule":     0.05,
    "chelator":           0.05,
    "porphyrin":          0.3,    # Fixed macrocycle size
    "crown_ether":        0.1,    # 12-crown-4 minimum
    "peptide":            0.3,
    "protein":            0.3,
    "antibody_CDR":       0.5,
    "aptamer":            0.5,
    "dnazyme":            0.5,
    "DNA_origami":        2.0,    # Can't position below helix diameter
    "MOF":                0.3,    # Minimum pore from smallest linkers
    "crystal":            0.1,
    "ion_exchange_resin":  0.2,
}

# Available donor atom types per material system
AVAILABLE_DONORS = {
    "small_molecule":     {"O_carboxylate", "O_hydroxyl", "O_ether", "O_carbonyl",
                           "N_amine", "N_pyridine", "N_imidazole", "N_amide",
                           "S_thiolate", "S_thioether", "P_phosphonate"},
    "chelator":           {"O_carboxylate", "O_hydroxyl", "O_ether", "O_phenolate",
                           "N_amine", "N_pyridine", "N_imidazole",
                           "S_thiolate", "S_thioether", "P_phosphonate"},
    "porphyrin":          {"N_pyrrole", "N_pyridine", "O_carboxylate"},  # Periphery can be functionalized
    "crown_ether":        {"O_ether", "N_amine", "S_thioether"},          # Aza/thia crowns
    "peptide":            {"O_carboxylate", "O_hydroxyl", "O_carbonyl",
                           "N_amine", "N_imidazole", "N_amide",
                           "S_thiolate", "S_thioether"},
    "protein":            {"O_carboxylate", "O_hydroxyl", "O_carbonyl",
                           "N_amine", "N_imidazole", "N_amide",
                           "S_thiolate", "S_thioether"},
    "antibody_CDR":       {"O_carboxylate", "O_hydroxyl", "O_carbonyl",
                           "N_amine", "N_imidazole", "N_amide",
                           "S_thiolate", "S_thioether"},
    "aptamer":            {"O_phosphate", "O_ribose", "N_base_ring",
                           "O_carbonyl"},                                   # No native S donors
    "dnazyme":            {"O_phosphate", "O_ribose", "N_base_ring",
                           "O_carbonyl"},
    "DNA_origami":        {"O_phosphate", "N_base_ring", "O_carbonyl"},    # Plus whatever is conjugated
    "MOF":                {"O_carboxylate", "O_hydroxyl", "N_pyridine",
                           "N_imidazole", "S_thiolate", "N_amine"},        # Depends on linker
    "crystal":            {"O_oxide", "O_hydroxyl", "S_sulfide"},
    "ion_exchange_resin": {"O_carboxylate", "O_sulfonate", "N_amine",
                           "O_phosphonate", "S_thiol"},
}

# Condition survival limits
CONDITION_LIMITS = {
    # (pH_min, pH_max, T_max_C, survives_strong_oxidant)
    "small_molecule":     (0, 14, 300, True),      # Depends on specific molecule
    "chelator":           (0, 14, 200, True),
    "porphyrin":          (1, 13, 200, False),      # Demetallation at extremes
    "crown_ether":        (1, 13, 150, True),
    "peptide":            (2, 12, 80, False),
    "protein":            (3, 11, 70, False),       # Most denature
    "antibody_CDR":       (4, 10, 60, False),
    "aptamer":            (4, 10, 70, False),       # Depurination at low pH
    "dnazyme":            (4, 10, 70, False),
    "DNA_origami":        (5, 9, 60, False),        # Needs Mg²⁺ buffer, narrow stability
    "MOF":                (1, 12, 300, True),       # Varies hugely; Zr-MOFs very stable
    "crystal":            (0, 14, 500, True),       # Inorganic crystals
    "ion_exchange_resin": (0, 14, 120, True),       # Polymer degradation at high T
}


@dataclass
class DisqualificationResult:
    """Result of running all gates for one realization type."""
    realization_type: str
    passed: bool
    reason: Optional[str] = None   # None if passed, explanation if failed


def check_scale_mismatch(
    realization_type: str,
    required_cavity_nm: float,
) -> Optional[str]:
    """Returns disqualification reason or None if passed."""
    min_cav = MIN_CAVITY.get(realization_type)
    max_cav = MAX_CAVITY.get(realization_type)
    if min_cav is None or max_cav is None:
        return f"Unknown realization type: {realization_type}"
    if required_cavity_nm < min_cav:
        return (
            f"Required cavity {required_cavity_nm:.2f} nm < minimum achievable "
            f"{min_cav:.2f} nm for {realization_type}"
        )
    if required_cavity_nm > max_cav:
        return (
            f"Required cavity {required_cavity_nm:.2f} nm > maximum achievable "
            f"{max_cav:.2f} nm for {realization_type}"
        )
    return None


def check_donor_availability(
    realization_type: str,
    required_donors: set[str],
) -> Optional[str]:
    """Returns disqualification reason or None if all required donors available."""
    available = AVAILABLE_DONORS.get(realization_type, set())
    missing = required_donors - available
    if missing:
        return (
            f"{realization_type} cannot natively present donor(s): "
            f"{', '.join(sorted(missing))}"
        )
    return None


def check_condition_compatibility(
    realization_type: str,
    pH: Optional[float] = None,
    temperature_C: Optional[float] = None,
    strong_oxidant: bool = False,
) -> Optional[str]:
    """Returns disqualification reason or None if conditions are survivable."""
    limits = CONDITION_LIMITS.get(realization_type)
    if limits is None:
        return f"Unknown realization type: {realization_type}"
    pH_min, pH_max, T_max, survives_ox = limits

    if pH is not None:
        if pH < pH_min:
            return (
                f"{realization_type} unstable at pH {pH:.1f} "
                f"(minimum survivable: {pH_min})"
            )
        if pH > pH_max:
            return (
                f"{realization_type} unstable at pH {pH:.1f} "
                f"(maximum survivable: {pH_max})"
            )
    if temperature_C is not None:
        if temperature_C > T_max:
            return (
                f"{realization_type} unstable at {temperature_C:.0f}°C "
                f"(maximum survivable: {T_max}°C)"
            )
    if strong_oxidant and not survives_ox:
        return (
            f"{realization_type} degraded by strong oxidants "
            f"(e.g. free chlorine, peroxide)"
        )
    return None


def run_all_gates(
    realization_type: str,
    required_cavity_nm: float,
    required_donors: set[str],
    pH: Optional[float] = None,
    temperature_C: Optional[float] = None,
    strong_oxidant: bool = False,
) -> DisqualificationResult:
    """Run all disqualification gates. First failure stops."""

    # Gate 1: Scale mismatch
    reason = check_scale_mismatch(realization_type, required_cavity_nm)
    if reason:
        return DisqualificationResult(realization_type, passed=False, reason=reason)

    # Gate 2: Donor availability
    reason = check_donor_availability(realization_type, required_donors)
    if reason:
        return DisqualificationResult(realization_type, passed=False, reason=reason)

    # Gate 3: Condition compatibility
    reason = check_condition_compatibility(
        realization_type, pH, temperature_C, strong_oxidant
    )
    if reason:
        return DisqualificationResult(realization_type, passed=False, reason=reason)

    return DisqualificationResult(realization_type, passed=True)


# All realization types the ranker knows about
ALL_REALIZATION_TYPES = list(PRECISION_LIMITS.keys())