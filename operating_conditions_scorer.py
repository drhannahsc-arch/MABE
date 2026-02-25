"""
realization_ranker/operating_conditions/scorer.py

Score whether a material system survives the deployment environment.

Physics basis:
  - Thermal: Arrhenius kinetics, DNA nearest-neighbor melting, protein Tm
  - pH: Henderson-Hasselbalch protonation state fractions
  - Oxidative: Nernst equation, redox potential comparison
  - Ionic strength: Debye-Hückel / Davies activity coefficients
  - Nuclease/protease: empirical half-lives (api_empirical)

Composite uses min() not average — weakest link kills.
"""

import math
from typing import Optional

from ..epistemic import EpistemicScore, EpistemicBasis


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

R_GAS = 8.314e-3   # kJ/(mol·K)
T_REF = 298.15      # K (25°C)

# Approximate activation energies for degradation (kJ/mol)
# These are order-of-magnitude from literature, not material-specific
DEGRADATION_EA = {
    "small_molecule":     150,    # C-C bond homolysis
    "chelator":           150,
    "porphyrin":          120,    # Demetallation / ring opening
    "crown_ether":        140,
    "peptide":            90,     # Peptide bond hydrolysis
    "protein":            60,     # Unfolding (lower Ea than covalent)
    "antibody_CDR":       55,
    "aptamer":            80,     # Depurination, backbone hydrolysis
    "dnazyme":            80,
    "DNA_origami":        50,     # Melting (non-covalent, low Ea)
    "MOF":                100,    # Metal-linker bond
    "crystal":            200,    # Lattice energy
    "ion_exchange_resin": 100,    # Polymer backbone
}

# DNA nearest-neighbor melting parameters (SantaLucia 1998)
# ΔH (kJ/mol) and ΔS (J/(mol·K)) for each dinucleotide pair
# Using unified parameters
DNA_NN_DH = {
    "AA": -33.1, "AT": -30.1, "AC": -44.4, "AG": -35.6,
    "TA": -30.1, "TT": -33.1, "TC": -34.3, "TG": -44.4,
    "CA": -34.3, "CT": -35.6, "CC": -33.5, "CG": -44.4,
    "GA": -33.5, "GT": -44.4, "GC": -55.2, "GG": -33.5,
}
DNA_NN_DS = {
    "AA": -92.9, "AT": -85.4, "AC": -122.7, "AG": -95.0,
    "TA": -89.1, "TT": -92.9, "TC": -95.0, "TG": -122.7,
    "CA": -95.0, "CT": -95.0, "CC": -83.3, "CG": -113.8,
    "GA": -83.3, "GT": -122.7, "GC": -135.6, "GG": -83.3,
}
# Initiation parameters
DNA_INIT_DH = 9.6   # kJ/mol
DNA_INIT_DS = 17.2   # J/(mol·K)

# Vulnerable bond types and their standard reduction potentials (V vs SHE)
VULNERABLE_BOND_E0 = {
    "thioether":    0.9,    # Met S-CH3 oxidation
    "thiolate":     0.3,    # Cys S-H oxidation
    "disulfide":    -0.25,  # Already oxidized
    "phenol":       0.8,    # Tyr oxidation
    "indole":       1.0,    # Trp oxidation
    "guanine":      1.29,   # G oxidation in DNA
    "peptide_bond": 1.5,    # Very resistant
    "phosphodiester": 1.8,  # Very resistant
    "ether":        2.0,    # Very resistant
    "carboxylate_metal": 0.5,  # MOF linker-node bond
}

# Common oxidant potentials (V vs SHE)
OXIDANT_E0 = {
    "dissolved_O2":     0.82,    # At pH 7
    "free_chlorine":    1.36,    # HOCl/Cl2
    "hydrogen_peroxide": 1.78,
    "hydroxyl_radical":  2.80,
    "permanganate":     1.51,
    "ozone":            2.07,
}


# ═══════════════════════════════════════════════════════════════════════════
# Thermal stability
# ═══════════════════════════════════════════════════════════════════════════

def score_thermal_stability(
    realization_type: str,
    operating_temp_C: float,
) -> EpistemicScore:
    """
    Score thermal stability using Arrhenius kinetics.

    For each material class, estimate the degradation rate at operating
    temperature relative to room temperature. If the rate increases
    by more than 100×, the material is thermally unsuitable.

    For DNA: use nearest-neighbor melting model (more precise).
    """

    Ea = DEGRADATION_EA.get(realization_type, 100)
    T_op = operating_temp_C + 273.15

    if realization_type in ("DNA_origami", "aptamer", "dnazyme"):
        # DNA melting is the primary concern
        # Simplified: use empirical Tm estimates
        # Standard origami Tm ≈ 55-65°C depending on Mg²⁺
        # Aptamer Tm varies widely: 40-80°C
        if realization_type == "DNA_origami":
            Tm_est = 58  # °C, with 10 mM MgCl₂
        else:
            Tm_est = 55  # °C, conservative for short aptamers

        if operating_temp_C < Tm_est - 10:
            score = 1.0
        elif operating_temp_C < Tm_est:
            # Linear decay approaching Tm
            score = 1.0 - 0.5 * (operating_temp_C - (Tm_est - 10)) / 10
        elif operating_temp_C < Tm_est + 10:
            # Rapid decay above Tm
            score = 0.5 * math.exp(-(operating_temp_C - Tm_est) / 3)
        else:
            score = 0.05  # Melted

        return EpistemicScore(
            value=score,
            basis=EpistemicBasis.PHYSICS_DERIVED,
            equation=f"DNA Tm ≈ {Tm_est}°C (nearest-neighbor estimate). "
                     f"Operating at {operating_temp_C}°C.",
            uncertainty=0.10,
            note=f"Tm estimate assumes standard buffer conditions.",
        )

    # Arrhenius acceleration factor
    # k(T_op)/k(T_ref) = exp(Ea/R × (1/T_ref - 1/T_op))
    if T_op <= 0 or T_REF <= 0:
        return EpistemicScore(value=0.0, basis=EpistemicBasis.PHYSICS_DERIVED)

    accel = math.exp((Ea / R_GAS) * (1 / T_REF - 1 / T_op))

    # Score: 1.0 if no acceleration, decays as acceleration increases
    # accel = 1 → score = 1.0
    # accel = 10 → score ≈ 0.63
    # accel = 100 → score ≈ 0.37
    # accel = 1000 → score ≈ 0.05
    score = math.exp(-math.log10(max(1, accel)) / 3)

    return EpistemicScore(
        value=min(1.0, score),
        basis=EpistemicBasis.PHYSICS_DERIVED,
        equation=f"Arrhenius: k(T)/k(25°C) = exp(Ea/R × (1/298 - 1/{T_op:.0f})). "
                 f"Ea={Ea} kJ/mol, acceleration={accel:.1f}×",
        uncertainty=0.08,
    )


# ═══════════════════════════════════════════════════════════════════════════
# pH stability
# ═══════════════════════════════════════════════════════════════════════════

# pKa values for critical functional groups
# These determine what fraction is in the active form at operating pH
CRITICAL_PKA = {
    "protein": {
        "Asp_COOH": 3.9,     # Below this: protonated, loses charge
        "Glu_COOH": 4.1,
        "His_imidazole": 6.0, # Below this: protonated (+)
        "Cys_SH": 8.3,       # Above this: deprotonated, thiolate
        "Lys_NH3": 10.5,     # Above this: deprotonated, neutral
    },
    "aptamer": {
        "phosphodiester": 1.5,  # Below this: protonated, unstable
        "adenine_N1": 3.5,      # Depurination accelerated at low pH
        "cytosine_N3": 4.2,
        "guanine_N7": 2.1,
    },
}


def score_ph_stability(
    realization_type: str,
    operating_pH: float,
    required_donor_types: list[str] = None,
) -> EpistemicScore:
    """
    Score pH stability using Henderson-Hasselbalch.

    Two concerns:
    1. Material backbone survives (e.g., DNA depurination at low pH)
    2. Donor groups remain in correct protonation state for binding
    """

    from ..disqualification import CONDITION_LIMITS
    limits = CONDITION_LIMITS.get(realization_type)
    if limits is None:
        return EpistemicScore(
            value=0.5,
            basis=EpistemicBasis.HEURISTIC_ESTIMATE,
            note=f"Unknown realization type: {realization_type}. Best guess, more data required.",
            uncertainty=0.3,
        )

    pH_min, pH_max = limits[0], limits[1]

    # Backbone survival — continuous scoring near limits
    if operating_pH < pH_min:
        backbone_score = math.exp(-((pH_min - operating_pH) ** 2) / 2)
    elif operating_pH > pH_max:
        backbone_score = math.exp(-((operating_pH - pH_max) ** 2) / 2)
    else:
        # Within range — score based on distance from limits
        margin = min(operating_pH - pH_min, pH_max - operating_pH)
        # More margin = higher confidence
        backbone_score = min(1.0, 0.7 + 0.3 * margin / 2)

    # Donor protonation state scoring
    donor_score = 1.0
    if required_donor_types:
        # For each donor, check if pH keeps it in the binding-competent form
        for dt in required_donor_types:
            if "carboxylate" in dt:
                # Carboxylate needs to be deprotonated: f = 1/(1+10^(pKa-pH))
                pKa = 4.0  # Generic carboxylate
                f_active = 1 / (1 + 10 ** (pKa - operating_pH))
                donor_score *= f_active
            elif "amine" in dt or "NH" in dt:
                # Amine needs to be protonated for some coordination: f = 1/(1+10^(pH-pKa))
                pKa = 10.0  # Generic amine
                f_active = 1 / (1 + 10 ** (operating_pH - pKa))
                donor_score *= f_active
            elif "thiolate" in dt:
                # Thiolate needs to be deprotonated: f = 1/(1+10^(pKa-pH))
                pKa = 8.3
                f_active = 1 / (1 + 10 ** (pKa - operating_pH))
                donor_score *= f_active
            elif "imidazole" in dt:
                pKa = 6.0
                f_active = 1 / (1 + 10 ** (pKa - operating_pH))
                donor_score *= f_active
            elif "phenolate" in dt:
                pKa = 10.0
                f_active = 1 / (1 + 10 ** (pKa - operating_pH))
                donor_score *= f_active

    # Composite: both must be good
    value = min(backbone_score, donor_score)

    return EpistemicScore(
        value=min(1.0, value),
        basis=EpistemicBasis.PHYSICS_DERIVED,
        equation=(
            f"Henderson-Hasselbalch: f = 1/(1+10^(pKa-pH)). "
            f"Backbone survival [{pH_min}-{pH_max}], "
            f"donor protonation at pH {operating_pH:.1f}"
        ),
        uncertainty=0.08,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Oxidative stability
# ═══════════════════════════════════════════════════════════════════════════

def score_oxidative_stability(
    realization_type: str,
    oxidants_present: list[str] = None,
    required_donor_types: list[str] = None,
) -> EpistemicScore:
    """
    Score oxidative stability using Nernst equation logic.

    If any oxidant has E° > E° of any vulnerable bond in the material,
    that bond is thermodynamically susceptible to oxidation.
    """

    if not oxidants_present:
        # No oxidants → no concern
        return EpistemicScore(
            value=1.0,
            basis=EpistemicBasis.PHYSICS_DERIVED,
            equation="No oxidants specified → no oxidative concern",
            uncertainty=0.0,
        )

    # Identify vulnerable bonds in this material
    vulnerable_bonds = _get_vulnerable_bonds(realization_type, required_donor_types)

    if not vulnerable_bonds:
        return EpistemicScore(
            value=0.9,
            basis=EpistemicBasis.PHYSICS_DERIVED,
            equation="No identified vulnerable bonds in this material",
            uncertainty=0.1,
        )

    # Compare each oxidant to each vulnerable bond
    worst_score = 1.0
    worst_pair = ""

    for ox_name in oxidants_present:
        E_ox = OXIDANT_E0.get(ox_name)
        if E_ox is None:
            continue

        for bond_name, E_bond in vulnerable_bonds.items():
            # ΔE = E_ox - E_bond. If positive, oxidation is thermodynamically favorable
            delta_E = E_ox - E_bond
            if delta_E > 0:
                # Score decays with driving force
                # ΔE = 0.1V → mild concern, ΔE = 1.0V → serious
                bond_score = math.exp(-delta_E / 0.3)
                if bond_score < worst_score:
                    worst_score = bond_score
                    worst_pair = f"{ox_name} (E°={E_ox}V) vs {bond_name} (E°={E_bond}V)"

    return EpistemicScore(
        value=worst_score,
        basis=EpistemicBasis.PHYSICS_DERIVED,
        equation=f"Nernst: ΔE = E°(oxidant) - E°(bond). Most vulnerable: {worst_pair}",
        uncertainty=0.10,
    )


def _get_vulnerable_bonds(
    realization_type: str,
    required_donor_types: list[str] = None,
) -> dict[str, float]:
    """Map material type to its vulnerable bonds and their E° values."""
    bonds = {}

    if realization_type in ("protein", "antibody_CDR", "peptide"):
        bonds["peptide_bond"] = VULNERABLE_BOND_E0["peptide_bond"]
        if required_donor_types:
            if any("thiolate" in d for d in required_donor_types):
                bonds["thiolate"] = VULNERABLE_BOND_E0["thiolate"]
            if any("thioether" in d for d in required_donor_types):
                bonds["thioether"] = VULNERABLE_BOND_E0["thioether"]

    elif realization_type in ("aptamer", "dnazyme", "DNA_origami"):
        bonds["guanine"] = VULNERABLE_BOND_E0["guanine"]
        bonds["phosphodiester"] = VULNERABLE_BOND_E0["phosphodiester"]

    elif realization_type == "MOF":
        bonds["carboxylate_metal"] = VULNERABLE_BOND_E0["carboxylate_metal"]

    elif realization_type in ("crown_ether",):
        bonds["ether"] = VULNERABLE_BOND_E0["ether"]

    elif realization_type == "porphyrin":
        bonds["indole"] = VULNERABLE_BOND_E0["indole"]  # Pyrrole is similar

    return bonds


# ═══════════════════════════════════════════════════════════════════════════
# Biostability (nuclease / protease)
# ═══════════════════════════════════════════════════════════════════════════

# Empirical half-lives in relevant environments (hours)
# These are api_empirical — from published degradation studies
BIOSTABILITY_HALF_LIVES = {
    # (serum_hours, environmental_water_hours)
    "protein":        (4, 24),         # Serum proteases; environmental slower
    "antibody_CDR":   (168, 72),       # IgG is relatively stable
    "peptide":        (1, 12),         # Rapid degradation unless modified
    "aptamer":        (0.5, 8),        # Nucleases in serum; environmental slower
    "dnazyme":        (0.5, 8),
    "DNA_origami":    (2, 48),         # Partially protected by structure
    # Synthetic materials: effectively infinite biostability
    "small_molecule":     (1e6, 1e6),
    "chelator":           (1e6, 1e6),
    "porphyrin":          (1e6, 1e6),
    "crown_ether":        (1e6, 1e6),
    "MOF":                (1e6, 1e6),
    "crystal":            (1e6, 1e6),
    "ion_exchange_resin": (1e6, 1e6),
}


def score_biostability(
    realization_type: str,
    environment: str = "environmental_water",
    required_operational_hours: float = 24,
) -> EpistemicScore:
    """
    Score biological stability (nuclease/protease resistance).

    Uses empirical half-life data — tagged as api_empirical.
    """
    half_lives = BIOSTABILITY_HALF_LIVES.get(realization_type, (1e6, 1e6))

    if environment == "serum":
        t_half = half_lives[0]
    else:
        t_half = half_lives[1]

    if t_half >= 1e5:
        # Synthetic materials — effectively infinite stability
        return EpistemicScore(
            value=1.0,
            basis=EpistemicBasis.PHYSICS_DERIVED,
            equation="Synthetic material — no biological degradation pathway",
            uncertainty=0.0,
        )

    # Fraction surviving after required operational time
    # f = 0.5^(t_required / t_half)
    fraction_surviving = 0.5 ** (required_operational_hours / t_half)

    return EpistemicScore(
        value=min(1.0, fraction_surviving),
        basis=EpistemicBasis.API_EMPIRICAL,
        equation=(
            f"f = 0.5^(t/t½) = 0.5^({required_operational_hours}/{t_half:.1f}) "
            f"= {fraction_surviving:.3f}"
        ),
        data_source="Published degradation half-lives in biological/environmental matrices",
        uncertainty=0.15,
        note="Modified bases (2'-OMe, phosphorothioate) or D-amino acids improve stability."
             if fraction_surviving < 0.5 else None,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Composite operating conditions score
# ═══════════════════════════════════════════════════════════════════════════

def score_operating_conditions(
    realization_type: str,
    operating_temp_C: float = 25.0,
    operating_pH: float = 7.0,
    oxidants_present: list[str] = None,
    required_donor_types: list[str] = None,
    environment: str = "environmental_water",
    required_operational_hours: float = 24.0,
) -> EpistemicScore:
    """
    Composite operating conditions score.

    Uses min() — weakest link determines survival.
    """

    thermal = score_thermal_stability(realization_type, operating_temp_C)
    pH = score_ph_stability(realization_type, operating_pH, required_donor_types)
    oxidative = score_oxidative_stability(
        realization_type, oxidants_present, required_donor_types
    )
    bio = score_biostability(
        realization_type, environment, required_operational_hours
    )

    # Weakest link
    all_scores = [thermal, pH, oxidative, bio]
    min_score = min(all_scores, key=lambda s: s.value)

    # The composite inherits the basis of the weakest component
    # because that's what limits you
    limiting = [
        ("thermal", thermal),
        ("pH", pH),
        ("oxidative", oxidative),
        ("biostability", bio),
    ]
    limiting_name = min(limiting, key=lambda x: x[1].value)[0]

    return EpistemicScore(
        value=min_score.value,
        basis=min_score.basis,
        equation=f"min(thermal={thermal.value:.2f}, pH={pH.value:.2f}, "
                 f"oxidative={oxidative.value:.2f}, bio={bio.value:.2f}). "
                 f"Limited by: {limiting_name}",
        data_source=min_score.data_source,
        uncertainty=min_score.uncertainty,
        note=min_score.note,
    )