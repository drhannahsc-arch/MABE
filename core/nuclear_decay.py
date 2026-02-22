"""
core/nuclear_decay.py — Sprint 29b: Nuclear Decay Chains + Radionuclide Routing

Models decay chains for nuclear waste targets. Daughter products have
different chemistry than parents — a binder designed for U-238 may
also capture Th-234 (different charge, size, HSAB class) or may let
daughters escape. Also handles Cs-137, Sr-90, Ra-226 for remediation.

Physics:
  Decay: N(t) = N₀ × exp(-λt), λ = ln(2)/t½
  Activity: A = λN (Bq)
  Daughter ingrowth: N_d(t) = (λ_p/(λ_d-λ_p)) × N₀ × (exp(-λ_p×t) - exp(-λ_d×t))
"""
from dataclasses import dataclass
import math


@dataclass
class Radionuclide:
    """Properties of a radioactive isotope relevant to binder design."""
    isotope: str                # "U-238", "Cs-137"
    element: str                # "U", "Cs"
    mass_number: int
    half_life_s: float
    half_life_display: str      # Human-readable
    decay_mode: str             # "alpha", "beta", "gamma", "EC", "IT"
    daughter: str               # Daughter isotope
    daughter_element: str
    daughter_charge: int        # Typical aqueous charge
    daughter_hsab: str          # "hard", "borderline", "soft"
    daughter_needs_separate_binder: bool
    specific_activity_bq_g: float  # Activity per gram
    dose_rate_factor: float     # Relative radiotoxicity
    notes: str = ""

@dataclass
class DecayChainAnalysis:
    """Full decay chain analysis for binder design."""
    parent: str
    chain: list                 # List of Radionuclide objects in chain
    total_species_to_capture: int
    chemistry_changes: list     # List of (parent→daughter) chemistry descriptions
    binder_strategy: str        # "single_binder", "multi_binder", "sequential"
    critical_daughters: list    # Daughters that escape current binder design
    time_to_secular_equilibrium_s: float
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# DECAY CHAIN DATABASE
# Relevant chains for environmental/nuclear remediation
# ═══════════════════════════════════════════════════════════════════════════

_YEAR_S = 3.156e7
_DAY_S = 86400
_HOUR_S = 3600
_MIN_S = 60

_DECAY_CHAINS = {
    "U-238": [
        ("U-238",  "U",  238, 4.47e9*_YEAR_S, "4.47 Gyr",  "alpha", "Th-234", "Th", 4, "hard", True,
         "UO2²⁺ (actinyl) → Th⁴⁺ (hard tetravalent): COMPLETELY different chemistry"),
        ("Th-234", "Th", 234, 24.1*_DAY_S,     "24.1 days", "beta",  "Pa-234m","Pa", 5, "hard", True,
         "Th⁴⁺ → Pa⁵⁺: both hard but different charge/size"),
        ("Pa-234m","Pa", 234, 1.17*_MIN_S,      "1.17 min",  "beta",  "U-234",  "U",  6, "hard", False,
         "Pa⁵⁺ → UO2²⁺: returns to uranium chemistry (rebindable)"),
        ("U-234",  "U",  234, 2.46e5*_YEAR_S,   "246 kyr",   "alpha", "Th-230", "Th", 4, "hard", True,
         "UO2²⁺ → Th⁴⁺ again"),
    ],
    "Cs-137": [
        ("Cs-137", "Cs", 137, 30.17*_YEAR_S,    "30.2 yr",   "beta",  "Ba-137m","Ba", 2, "hard", True,
         "Cs⁺ (large alkali, crown ether target) → Ba²⁺ (alkaline earth): "
         "different charge, different selectivity. Zeolite captures both."),
    ],
    "Sr-90": [
        ("Sr-90",  "Sr", 90,  28.8*_YEAR_S,     "28.8 yr",   "beta",  "Y-90",   "Y",  3, "hard", True,
         "Sr²⁺ → Y³⁺: charge change from +2 to +3. Crown ether loses selectivity."),
        ("Y-90",   "Y",  90,  64.0*_HOUR_S,      "64 hr",     "beta",  "Zr-90",  "Zr", 4, "hard", True,
         "Y³⁺ → Zr⁴⁺ (stable): hard, highly charged, forms hydroxides rapidly"),
    ],
    "Ra-226": [
        ("Ra-226", "Ra", 226, 1600*_YEAR_S,     "1600 yr",   "alpha", "Rn-222", "Rn", 0, "none", True,
         "Ra²⁺ → Rn (noble gas): ESCAPES any binder. Must contain with housing."),
        ("Rn-222", "Rn", 222, 3.82*_DAY_S,      "3.82 days", "alpha", "Po-218", "Po", 4, "soft", True,
         "Rn → Po: gas → soft metal. Fundamentally different binding strategy."),
    ],
    "I-131": [
        ("I-131",  "I",  131, 8.02*_DAY_S,      "8.02 days", "beta",  "Xe-131", "Xe", 0, "none", True,
         "I⁻ → Xe (noble gas): daughter escapes. Short t½ = decays away."),
    ],
    "Co-60": [
        ("Co-60",  "Co", 60,  5.27*_YEAR_S,     "5.27 yr",   "beta",  "Ni-60",  "Ni", 2, "borderline", False,
         "Co²⁺ → Ni²⁺: similar coordination chemistry. Same binder likely works."),
    ],
    "Tc-99": [
        ("Tc-99",  "Tc", 99,  2.11e5*_YEAR_S,   "211 kyr",   "beta",  "Ru-99",  "Ru", 3, "borderline", True,
         "TcO₄⁻ (pertechnetate, anion) → Ru³⁺ (cation): charge SIGN reversal"),
    ],
    "Am-241": [
        ("Am-241", "Am", 241, 432.2*_YEAR_S,    "432 yr",    "alpha", "Np-237", "Np", 5, "hard", True,
         "Am³⁺ → NpO₂⁺: actinide → actinyl. Geometry and charge change."),
    ],
    "Pu-239": [
        ("Pu-239", "Pu", 239, 2.41e4*_YEAR_S,   "24.1 kyr",  "alpha", "U-235",  "U",  6, "hard", False,
         "Pu⁴⁺ → UO₂²⁺: both captured by hard-acid binders."),
    ],
}


def get_radionuclide(isotope):
    """Get properties of a specific radionuclide."""
    for chain_key, chain in _DECAY_CHAINS.items():
        for entry in chain:
            if entry[0] == isotope:
                iso, elem, mass, t_half, t_disp, mode, daughter, d_elem, d_charge, d_hsab, d_sep, notes = entry
                lam = math.log(2) / t_half if t_half > 0 else 0
                specific = lam * 6.022e23 / (mass * 1e-3) if mass > 0 else 0
                dose_factor = 1.0
                if mode == "alpha": dose_factor = 20.0  # Alpha = 20× more damaging
                return Radionuclide(
                    isotope=iso, element=elem, mass_number=mass,
                    half_life_s=t_half, half_life_display=t_disp,
                    decay_mode=mode, daughter=daughter,
                    daughter_element=d_elem, daughter_charge=d_charge,
                    daughter_hsab=d_hsab,
                    daughter_needs_separate_binder=d_sep,
                    specific_activity_bq_g=specific,
                    dose_rate_factor=dose_factor,
                    notes=notes,
                )
    return None


def analyze_decay_chain(parent_isotope):
    """Analyze full decay chain for binder design implications."""
    chain_data = _DECAY_CHAINS.get(parent_isotope)
    if chain_data is None:
        return None

    chain = []
    chemistry_changes = []
    critical_daughters = []
    total_species = 1  # The parent

    for entry in chain_data:
        rn = get_radionuclide(entry[0])
        if rn:
            chain.append(rn)
            if rn.daughter_needs_separate_binder:
                chemistry_changes.append(
                    f"{rn.isotope}({rn.element}{'+' if rn.daughter_charge > 0 else ''}) → "
                    f"{rn.daughter}({rn.daughter_element}{rn.daughter_charge}+): {rn.notes[:80]}")
                critical_daughters.append(rn.daughter)
                total_species += 1

    # Strategy
    if len(critical_daughters) == 0:
        strategy = "single_binder"
    elif any(d_elem in ("Rn", "Xe", "Kr") for entry in chain_data
             for d_elem in [entry[7]] if entry[10]):
        strategy = "containment_housing"  # Noble gas daughter = can't bind
    elif total_species <= 2:
        strategy = "multi_binder"
    else:
        strategy = "sequential"

    # Secular equilibrium: ~7 × t½ of longest-lived daughter
    daughter_halflives = [entry[3] for entry in chain_data if entry[10]]
    t_equil = max(daughter_halflives) * 7 if daughter_halflives else 0

    notes = ""
    gas_daughters = [entry[7] for entry in chain_data if entry[7] in ("Rn", "Xe")]
    if gas_daughters:
        notes = f"WARNING: Decay produces noble gas ({', '.join(gas_daughters)}) — will escape any binder"

    return DecayChainAnalysis(
        parent=parent_isotope, chain=chain,
        total_species_to_capture=total_species,
        chemistry_changes=chemistry_changes,
        binder_strategy=strategy,
        critical_daughters=critical_daughters,
        time_to_secular_equilibrium_s=t_equil,
        notes=notes,
    )

