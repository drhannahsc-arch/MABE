"""
interface_extraction.py — MABE Environment-Aware Competition Module (v2)

Provides ΔG correction for extracting metals from oil-water interfaces
in systems containing naphthenic acid fraction compounds (NAFCs).

Key correction from v1:
  v1 used acetate as sole naphthenate proxy.
  v2 uses Ox-class speciation-weighted mixture:
    O2 (monocarboxylic, ~50%) → acetate proxy, log K 0.5-3.7
    O3 (hydroxy-acid, ~20%)   → glycolate proxy, log K 0.7-4.4
    O4 (dicarboxylic, ~15%)   → oxalate proxy, log K 3.0-7.5  ← chelate effect
    O5+ (poly-oxy, ~10%)      → citrate proxy, log K 3.5-11.5 ← chelate effect
    O8 (ARN tetraacid, ~1%)   → estimated from scaffold chelation

  The O4+ fraction creates 3-8 log K stronger competition than acetate alone.
  Ignoring it underestimates the extraction penalty by that amount for
  ~25-30% of the NAFC pool.

Data sources:
  NIST SRD 46 — metal-acetate, glycolate, oxalate, malonate, citrate log K
  Orbitrap MS literature — OSPW Ox-class distributions
  Chain-length invariance validation — confirmed C2-C8 range ±0.11 log K

Usage:
  from interface_extraction import InterfaceEnvironment, effective_log_K

  env = InterfaceEnvironment.ospw_fresh()
  log_K_eff = effective_log_K(18.0, 'Pb2+', env)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

# ═══════════════════════════════════════════════════════════════════════════
# Ox-CLASS PROXY TABLES (from NIST SRD 46 calibration data)
# ═══════════════════════════════════════════════════════════════════════════
#
# Each table: metal → log K₁ for the proxy ligand.
# These are thermodynamic constants at I=0.1M, 25°C.

# O2 proxy: Acetate (CH₃COO⁻), monodentate
# Chain-length invariant — validated across C1-C8, C2→C7 Δ = -0.00 ± 0.11
LOG_K_O2 = {
    'Ag+': 0.73, 'Al3+': 2.04, 'Ba2+': 0.44, 'Ca2+': 0.55,
    'Cd2+': 1.52, 'Ce3+': 1.91, 'Co2+': 0.86, 'Cu2+': 1.79,
    'Dy3+': 1.85, 'Eu3+': 2.13, 'Fe2+': 0.83, 'Fe3+': 3.60,
    'Gd3+': 2.02, 'Hg2+': 3.74, 'La3+': 1.80, 'Lu3+': 1.85,
    'Mg2+': 0.51, 'Mn2+': 0.80, 'Na+': -0.27, 'Nd3+': 2.10,
    'Ni2+': 0.88, 'Pb2+': 2.10, 'Sc3+': 3.48, 'Sr2+': 0.47,
    'Yb3+': 1.84, 'Zn2+': 1.07,
}

# O3 proxy: Glycolate (HOCH₂COO⁻), bidentate (COO⁻ + OH)
# True glycolate only (dent=3 entries), NOT diglycolic acid (dent=5).
# Values close to acetate for most divalents: the -OH barely participates
# at neutral pH unless metal is hard + high charge.
LOG_K_O3 = {
    'Ba2+': 0.66, 'Ca2+': 1.11, 'Co2+': 1.66, 'Cu2+': 2.40,
    'Mg2+': 0.92, 'Mn2+': 1.11, 'Sc3+': 4.40, 'Sr2+': 0.80,
    'Zn2+': 1.95,
    # From lactate where glycolate unavailable:
    'La3+': 2.51, 'Ce3+': 2.56, 'Eu3+': 2.90, 'Gd3+': 2.91,
    'Nd3+': 2.70, 'Yb3+': 3.25,
}

# O4 proxy: Oxalate (⁻OOC-COO⁻), bidentate chelate forming 5-member ring
# THIS IS WHERE THE CHELATE EFFECT KICKS IN.
# Cu: acetate 1.8 → oxalate 4.8  (+3.0 log K)
# Fe³⁺: acetate 3.6 → oxalate 7.5 (+3.9 log K)
LOG_K_O4 = {
    'Al3+': 6.10, 'Ca2+': 3.00, 'Cd2+': 3.90, 'Co2+': 4.70,
    'Cr3+': 5.50, 'Cu2+': 4.80, 'Fe2+': 3.10, 'Fe3+': 7.50,
    'Hg2+': 4.70, 'La3+': 5.00, 'Mg2+': 3.40, 'Mn2+': 3.90,
    'Ni2+': 5.20, 'Pb2+': 4.90, 'Zn2+': 4.90,
}

# O5+ proxy: Citrate (3 carboxylates + 1 hydroxyl, tridentate chelate)
# The strongest simple carboxylate competition.
# Fe³⁺: citrate 11.5 — this is what locks iron in tailings.
LOG_K_O5 = {
    'Al3+': 9.60, 'Ca2+': 3.50, 'Cd2+': 3.80, 'Co2+': 5.00,
    'Cu2+': 6.10, 'Fe3+': 11.50, 'Gd3+': 7.60, 'La3+': 7.30,
    'Mg2+': 3.40, 'Mn2+': 3.70, 'Ni2+': 5.40, 'Pb2+': 4.10,
    'Zn2+': 5.00,
}

# O8 proxy: ARN tetraacid (C₈₀H₁₄₂O₈, 4× COOH on rigid scaffold)
# No direct calibration data. Estimated from scaffold chelation principles:
#   - 4 carboxylates with ~10Å spacing → can wrap divalent metal
#   - Macrocyclic-like preorganization on rigid hydrocarbon skeleton
#   - Estimated log K ≈ 2× oxalate log K + 1-2 (scaffold preorganization)
# This is the species responsible for calcium naphthenate deposition.
# Taylor & Chu 2018: Ca²⁺ binds specifically; not Mg²⁺, Sr²⁺.
LOG_K_O8_ESTIMATE = {
    'Ca2+': 8.0,   # Drives CaN deposition; must be high to precipitate
    'Cu2+': 12.0,  # ~2× oxalate + preorganization
    'Fe3+': 16.0,  # Very strong — chelate + hard acid
    'Pb2+': 11.0,  # Strong chelation
    'Zn2+': 10.0,
    'Ni2+': 11.0,
    'Cd2+': 9.0,
    'Mn2+': 8.0,
    'Co2+': 10.0,
    'Hg2+': 10.0,
    'Al3+': 14.0,
    'La3+': 12.0,
    'Mg2+': 6.0,   # Weaker — Mg²⁺ smaller, less preorganization benefit
}

# Mapping: Ox class → proxy table
OX_PROXY_TABLES = {
    'O2': LOG_K_O2,
    'O3': LOG_K_O3,
    'O4': LOG_K_O4,
    'O5+': LOG_K_O5,
    'O8': LOG_K_O8_ESTIMATE,
}

# Fallback: if a metal isn't in a proxy table, estimate from charge
def _fallback_logK(metal: str, ox_class: str) -> float:
    """Estimate log K when metal not in proxy table."""
    charge = int(metal[-2]) if len(metal) > 2 and metal[-2].isdigit() else 1
    base = {
        'O2': {1: 0.0, 2: 1.2, 3: 2.5, 4: 4.0},
        'O3': {1: 0.0, 2: 1.5, 3: 3.0, 4: 5.0},
        'O4': {1: 0.5, 2: 4.0, 3: 6.0, 4: 9.0},
        'O5+': {1: 0.5, 2: 4.5, 3: 8.0, 4: 12.0},
        'O8': {1: 1.0, 2: 9.0, 3: 13.0, 4: 18.0},
    }
    return base.get(ox_class, base['O2']).get(charge, 1.0)


def get_proxy_logK(metal: str, ox_class: str) -> float:
    """Look up log K for metal with given Ox-class proxy."""
    table = OX_PROXY_TABLES.get(ox_class, LOG_K_O2)
    if metal in table:
        return table[metal]
    return _fallback_logK(metal, ox_class)


# ═══════════════════════════════════════════════════════════════════════════
# logP MODEL
# ═══════════════════════════════════════════════════════════════════════════

LOGP_SLOPE = 0.390           # per carbon (RDKit validated, R²=1.000)
LOGP_INTERCEPT = -0.689      # offset
RING_CORRECTION = -0.30      # per naphthenic ring


# ═══════════════════════════════════════════════════════════════════════════
# NAFC SPECIATION PROFILES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class NAFCProfile:
    """
    Ox-class distribution of naphthenic acid fraction compounds.

    Fractions should sum to ~1.0. Each fraction represents the mole
    proportion of NAFCs in that Ox class that are available for metal
    binding at the interface.

    Sources:
      - Orbitrap MS: Headley et al. 2016, Barrow et al. 2015
      - FT-ICR MS: various OSPW characterization studies
      - Wetland aging studies show O2 decreases, O3+ increases over time
    """
    f_O2: float = 0.50   # Classical monocarboxylic NAs
    f_O3: float = 0.20   # Hydroxy-acids, keto-acids
    f_O4: float = 0.15   # Dicarboxylic acids (chelating!)
    f_O5: float = 0.10   # Tri-/poly-carboxylic + hydroxyl
    f_O8: float = 0.0    # ARN tetraacids — UNCALIBRATED, off by default
    # O8 note: ARN tetraacids (C₈₀H₁₄₂O₈) have no NIST calibration data.
    # Their log K estimates are speculative (±5 log K uncertainty).
    # At even 0.5% fraction, uncalibrated O8 values dominate the entire
    # mixture model because 10^12 × 0.005 >> 10^5 × 0.13.
    # Enable only for specific ARN/calcium naphthenate studies.
    # The O4 (oxalate) and O5+ (citrate) proxies are NIST-calibrated
    # and already capture the chelate effect missing from v1.

    @classmethod
    def fresh_tailings(cls):
        """Fresh OSPW — dominated by O2, little oxidation."""
        return cls(f_O2=0.55, f_O3=0.18, f_O4=0.15, f_O5=0.10, f_O8=0.0)

    @classmethod
    def aged_tailings(cls):
        """Aged/weathered OSPW — O2 decreases, O3-O6 increase."""
        return cls(f_O2=0.35, f_O3=0.25, f_O4=0.22, f_O5=0.16, f_O8=0.0)

    @classmethod
    def groundwater_bitumen(cls):
        """Bitumen-influenced groundwater — O2 and O4 dominant."""
        return cls(f_O2=0.40, f_O3=0.15, f_O4=0.27, f_O5=0.14, f_O8=0.0)

    @classmethod
    def clean(cls):
        """No NAFCs present."""
        return cls(f_O2=0.0, f_O3=0.0, f_O4=0.0, f_O5=0.0, f_O8=0.0)

    @property
    def fractions(self) -> Dict[str, float]:
        return {
            'O2': self.f_O2, 'O3': self.f_O3, 'O4': self.f_O4,
            'O5+': self.f_O5, 'O8': self.f_O8,
        }

    @property
    def total_fraction(self) -> float:
        return sum(self.fractions.values())


# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT MODEL
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class InterfaceEnvironment:
    """
    Operating environment for interface extraction scoring.

    The key addition over v1: `nafc_profile` specifies the Ox-class
    distribution of NAFCs, enabling speciation-weighted competition.
    """
    name: str = "generic"
    pH: float = 7.0
    na_avg_carbon: float = 14.0
    na_avg_rings: float = 2.0
    na_concentration_mg_L: float = 80.0
    na_pKa: float = 4.9
    nafc_profile: NAFCProfile = field(default_factory=NAFCProfile)
    ionic_strength_M: float = 0.05
    interface_fraction_default: float = 0.15
    metal_interface_fractions: Dict[str, float] = field(default_factory=dict)
    temperature_C: float = 25.0

    @classmethod
    def ospw_fresh(cls):
        """Fresh Oil Sands Process-Affected Water."""
        return cls(
            name="OSPW_fresh",
            pH=8.0,
            na_avg_carbon=14,
            na_avg_rings=2.5,
            na_concentration_mg_L=80,
            nafc_profile=NAFCProfile.fresh_tailings(),
            ionic_strength_M=0.05,
            metal_interface_fractions={
                'Cu2+': 0.25, 'Zn2+': 0.15, 'Pb2+': 0.20,
                'Cd2+': 0.20, 'Ni2+': 0.15, 'Fe3+': 0.05,
                'Ca2+': 0.30, 'Mn2+': 0.10, 'Co2+': 0.15,
                'Hg2+': 0.35, 'Al3+': 0.03, 'La3+': 0.10,
                'Mg2+': 0.25,
            },
            temperature_C=10,
        )

    @classmethod
    def ospw_aged(cls):
        """Aged/weathered OSPW — more oxidized NAFCs, stronger competition."""
        env = cls.ospw_fresh()
        env.name = "OSPW_aged"
        env.nafc_profile = NAFCProfile.aged_tailings()
        return env

    @classmethod
    def acid_rock_drainage(cls):
        """ARD — low pH, minimal organic."""
        return cls(
            name="ARD",
            pH=3.5,
            na_avg_carbon=10,
            na_avg_rings=1.0,
            na_concentration_mg_L=5,
            nafc_profile=NAFCProfile(f_O2=0.60, f_O3=0.15, f_O4=0.12,
                                     f_O5=0.06, f_O8=0.0),
            ionic_strength_M=0.02,
            metal_interface_fractions={
                'Cu2+': 0.05, 'Zn2+': 0.03, 'Pb2+': 0.05,
                'Cd2+': 0.03, 'Fe3+': 0.02, 'Al3+': 0.01, 'Mn2+': 0.02,
            },
            temperature_C=15,
        )

    @classmethod
    def clean_water(cls):
        """No organic phase."""
        return cls(
            name="clean_water",
            nafc_profile=NAFCProfile.clean(),
            na_concentration_mg_L=0,
            metal_interface_fractions={},
        )

    @property
    def na_logP(self) -> float:
        if self.na_avg_carbon == 0:
            return 0.0
        return LOGP_SLOPE * self.na_avg_carbon + LOGP_INTERCEPT + \
               RING_CORRECTION * self.na_avg_rings

    @property
    def na_fraction_deprotonated(self) -> float:
        return 1.0 / (1.0 + 10**(self.na_pKa - self.pH))


# ═══════════════════════════════════════════════════════════════════════════
# SPECIATION-WEIGHTED COMPETITION PENALTY
# ═══════════════════════════════════════════════════════════════════════════

def weighted_nafc_logK(metal: str, profile: NAFCProfile) -> float:
    """
    Compute the speciation-weighted effective log K for NAFC competition.

    This is NOT a simple weighted average of log K values.
    The correct thermodynamics: each Ox fraction independently competes,
    and the metal distributes according to relative affinities.

    The effective competition is dominated by the STRONGEST binder
    present in significant quantity, not the average.

    For a mixture of competitors at concentrations c_i with log K_i:
      fraction bound to competitor i ∝ c_i × K_i
      effective K_competition = Σ(f_i × K_i)  [in linear K space]
      effective log K = log10(Σ(f_i × 10^(log K_i)))

    This correctly captures: even 15% O4 at log K = 5 overwhelms
    50% O2 at log K = 1.8, because 0.15 × 10^5 >> 0.50 × 10^1.8
    """
    if profile.total_fraction == 0:
        return 0.0

    sum_fK = 0.0
    for ox_class, fraction in profile.fractions.items():
        if fraction <= 0:
            continue
        logK_i = get_proxy_logK(metal, ox_class)
        sum_fK += fraction * (10 ** logK_i)

    if sum_fK <= 0:
        return 0.0

    return np.log10(sum_fK)


def interface_penalty(metal: str, environment: InterfaceEnvironment) -> float:
    """
    Total log K penalty for extracting metal from NAFC-containing system.

    Components:
      1. Speciation-weighted NAFC competition (Ox-class mixture)
      2. pH correction (protonated NAs can't bind)
      3. Interface desorption (logP-dependent anchoring at oil-water boundary)
    """
    if environment.na_concentration_mg_L == 0:
        return 0.0

    # 1. Speciation-weighted competition
    logK_nafc = weighted_nafc_logK(metal, environment.nafc_profile)

    # 2. pH correction
    f_deprot = environment.na_fraction_deprotonated
    # All Ox classes require deprotonation. At low pH, none bind.
    # Note: O4+ dicarboxylic acids have pKa1 ~2-3, pKa2 ~4-5
    # So at pH 3.5: O2 monoacids are ~30% deprotonated,
    # but O4 diacids have first carboxylate ~70% deprotonated.
    # Using single pKa is conservative for O4+.
    logK_ph_corrected = logK_nafc * f_deprot

    # 3. Interface desorption
    f_int = environment.metal_interface_fractions.get(
        metal, environment.interface_fraction_default)
    logP = environment.na_logP
    desorption = f_int * logP * 0.10

    return logK_ph_corrected + desorption


def effective_log_K(
    log_K_binder: float,
    metal: str,
    environment: InterfaceEnvironment,
) -> float:
    """
    Compute effective log K for binder extraction in given environment.

    log K_effective = log K_binder - interface_penalty(metal, env)
    """
    penalty = interface_penalty(metal, environment)
    return log_K_binder - penalty


# ═══════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS & REPORTING
# ═══════════════════════════════════════════════════════════════════════════

def penalty_breakdown(metal: str, environment: InterfaceEnvironment) -> Dict:
    """Detailed breakdown of penalty components for diagnostics."""
    profile = environment.nafc_profile
    f_deprot = environment.na_fraction_deprotonated
    f_int = environment.metal_interface_fractions.get(
        metal, environment.interface_fraction_default)

    contributions = {}
    total_fK = 0.0
    for ox_class, fraction in profile.fractions.items():
        if fraction <= 0:
            continue
        logK_i = get_proxy_logK(metal, ox_class)
        fK = fraction * (10 ** logK_i)
        total_fK += fK
        contributions[ox_class] = {
            'fraction': fraction,
            'logK_proxy': logK_i,
            'fK_linear': fK,
        }

    for ox_class in contributions:
        contributions[ox_class]['pct_of_penalty'] = \
            100.0 * contributions[ox_class]['fK_linear'] / total_fK if total_fK > 0 else 0

    return {
        'metal': metal,
        'environment': environment.name,
        'weighted_logK_nafc': np.log10(total_fK) if total_fK > 0 else 0,
        'pH_factor': f_deprot,
        'interface_fraction': f_int,
        'desorption_penalty': f_int * environment.na_logP * 0.10,
        'total_penalty': interface_penalty(metal, environment),
        'class_contributions': contributions,
    }


def extraction_report(
    binder_log_K: Dict[str, float],
    environment: InterfaceEnvironment,
    verbose: bool = False,
) -> str:
    """Generate extraction feasibility report."""
    lines = []
    lines.append(f"Extraction Report: {environment.name}")
    lines.append(f"  pH={environment.pH}, NA={environment.na_concentration_mg_L} mg/L, "
                 f"avg C{environment.na_avg_carbon}")
    pf = environment.nafc_profile
    lines.append(f"  NAFC profile: O2={pf.f_O2:.0%} O3={pf.f_O3:.0%} "
                 f"O4={pf.f_O4:.0%} O5+={pf.f_O5:.0%} O8={pf.f_O8:.1%}")
    lines.append(f"  pH deprotonation: {environment.na_fraction_deprotonated:.3f}")
    lines.append("")

    lines.append(f"{'Metal':8s} {'logK_bind':>9s} {'logK_NAFC':>9s} "
                 f"{'Penalty':>7s} {'logK_eff':>8s} {'Dominant':10s} {'Verdict'}")
    lines.append("─" * 72)

    for metal in sorted(binder_log_K.keys()):
        logK_b = binder_log_K[metal]
        bd = penalty_breakdown(metal, environment)
        logK_eff = logK_b - bd['total_penalty']

        dominant = max(bd['class_contributions'].items(),
                       key=lambda x: x[1]['pct_of_penalty'])
        dom_str = f"{dominant[0]}({dominant[1]['pct_of_penalty']:.0f}%)"

        if logK_eff > 8:
            verdict = "STRONG"
        elif logK_eff > 4:
            verdict = "FEASIBLE"
        elif logK_eff > 1:
            verdict = "MARGINAL"
        else:
            verdict = "UNLIKELY"

        lines.append(f"{metal:8s} {logK_b:9.1f} {bd['weighted_logK_nafc']:9.1f} "
                     f"{bd['total_penalty']:7.1f} {logK_eff:8.1f} {dom_str:10s} {verdict}")

    if verbose:
        lines.append("")
        lines.append("Detailed breakdown (first metal):")
        first = sorted(binder_log_K.keys())[0]
        bd = penalty_breakdown(first, environment)
        for ox, info in sorted(bd['class_contributions'].items()):
            lines.append(f"  {ox:4s}: f={info['fraction']:.3f}, "
                         f"logK={info['logK_proxy']:.1f}, "
                         f"contributes {info['pct_of_penalty']:.1f}%")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

def self_test():
    print("Interface Extraction Module v2 — Self-Test")
    print("═" * 65)

    # 1. Clean water → zero penalty
    env_clean = InterfaceEnvironment.clean_water()
    assert interface_penalty('Pb2+', env_clean) == 0.0
    print("  ✓ Clean water: zero penalty")

    # 2. O4 chelate effect: weighted penalty >> acetate-only penalty
    profile_o2_only = NAFCProfile(f_O2=1.0, f_O3=0, f_O4=0, f_O5=0, f_O8=0)
    profile_mixed = NAFCProfile.fresh_tailings()

    logK_o2 = weighted_nafc_logK('Cu2+', profile_o2_only)
    logK_mixed = weighted_nafc_logK('Cu2+', profile_mixed)
    chelate_boost = logK_mixed - logK_o2
    assert chelate_boost > 1.5, f"Expected >1.5 log K chelate boost, got {chelate_boost:.2f}"
    print(f"  ✓ Chelate effect: O2-only logK={logK_o2:.2f}, "
          f"mixed logK={logK_mixed:.2f}, boost={chelate_boost:.2f}")

    # 3. Fe³⁺ should have highest NAFC penalty (strongest chelate binder)
    env = InterfaceEnvironment.ospw_fresh()
    metals = ['Cu2+', 'Pb2+', 'Zn2+', 'Fe3+', 'Ca2+']
    penalties = {m: weighted_nafc_logK(m, profile_mixed) for m in metals}
    assert penalties['Fe3+'] > penalties['Cu2+'], "Fe3+ should have higher NAFC logK than Cu2+"
    print(f"  ✓ Fe³⁺ NAFC logK ({penalties['Fe3+']:.2f}) > "
          f"Cu²⁺ ({penalties['Cu2+']:.2f})")

    # 4. Dominant class should be O4 or O5+ for transition metals (chelate effect)
    bd = penalty_breakdown('Cu2+', env)
    o4_pct = bd['class_contributions'].get('O4', {}).get('pct_of_penalty', 0)
    o5_pct = bd['class_contributions'].get('O5+', {}).get('pct_of_penalty', 0)
    o2_pct = bd['class_contributions'].get('O2', {}).get('pct_of_penalty', 0)
    print(f"  ✓ Cu²⁺ penalty shares: O2={o2_pct:.0f}%, O4={o4_pct:.0f}%, O5+={o5_pct:.0f}%")
    assert (o4_pct + o5_pct) > o2_pct, "O4+O5 should dominate over O2 for Cu2+"

    # 5. pH correction: ARD << OSPW
    env_ard = InterfaceEnvironment.acid_rock_drainage()
    pen_ospw = interface_penalty('Pb2+', env)
    pen_ard = interface_penalty('Pb2+', env_ard)
    assert pen_ard < pen_ospw
    print(f"  ✓ Pb²⁺ penalty: OSPW={pen_ospw:.1f}, ARD={pen_ard:.1f}")

    # 6. Aged OSPW > Fresh OSPW (more oxidized species)
    env_aged = InterfaceEnvironment.ospw_aged()
    pen_fresh = interface_penalty('Cu2+', env)
    pen_aged = interface_penalty('Cu2+', env_aged)
    assert pen_aged > pen_fresh, "Aged OSPW should have higher penalty"
    print(f"  ✓ Cu²⁺ penalty: fresh={pen_fresh:.1f}, aged={pen_aged:.1f}")

    # 7. Extraction report — weak binder
    print("\n" + "─" * 65)
    weak = {
        'Pb2+': 4.9, 'Cu2+': 9.1, 'Zn2+': 6.4,
        'Ca2+': 4.5, 'Hg2+': 4.7, 'Fe3+': 12.0,
    }
    print(extraction_report(weak, env, verbose=True))

    # 8. Same binder, strong chelator (EDTA-class)
    print("\n" + "─" * 65)
    strong = {
        'Pb2+': 18.0, 'Cu2+': 18.8, 'Zn2+': 16.5,
        'Ca2+': 10.7, 'Hg2+': 21.8, 'Fe3+': 25.1,
    }
    print(extraction_report(strong, env))

    # 9. v1 vs v2 comparison
    print("\n" + "─" * 65)
    print("v1 (acetate-only) vs v2 (Ox-weighted) penalty comparison:")
    print(f"  {'Metal':8s} {'v1(OAc)':>8s} {'v2(Ox)':>8s} {'Δ':>8s}")
    print("  " + "─" * 36)
    for m in ['Ca2+', 'Mn2+', 'Zn2+', 'Cu2+', 'Pb2+', 'Fe3+', 'Hg2+']:
        v1 = LOG_K_O2.get(m, 1.0) * env.na_fraction_deprotonated
        v1 += env.metal_interface_fractions.get(m, 0.15) * env.na_logP * 0.10
        v2 = interface_penalty(m, env)
        print(f"  {m:8s} {v1:8.1f} {v2:8.1f} {v2-v1:+8.1f}")

    print("\n" + "═" * 65)
    print("All self-tests passed.")


if __name__ == "__main__":
    self_test()