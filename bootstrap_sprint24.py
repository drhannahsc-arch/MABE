"""
MABE Platform - Sprint 23+24 Bootstrap
Sprint 23: Cooperativity + Multi-Site Coupling
Sprint 24: Mass Transport + Diffusion Coupling

Requires Sprints 16-22 in place.
Run: python bootstrap_sprint23_24.py
Then: python tests/test_sprint23_24.py
"""
import os

def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Created: {path}")

print("\n\U0001f528 MABE Sprint 23+24 \u2014 System-Level Prediction\n")

write_file("core/cooperativity.py", '''\
"""
core/cooperativity.py — Sprint 23: Cooperativity + Multi-Site Coupling

Models interactions between multiple binding sites on a scaffold.
Electrostatic site-site repulsion, Hill coefficient cooperativity,
loading-dependent selectivity, and total capacity prediction.

Physics:
  ΔG_site_repulsion = z₁z₂e² / (4πε₀εᵣ × r_site)  (Coulomb between bound ions)
  K_apparent = K_intrinsic^n × [cooperativity_factor]  (Hill-like)
  Capacity = n_sites × site_quality × loading_efficiency
"""
from dataclasses import dataclass, field
import math


@dataclass
class CooperativityResult:
    """Multi-site coupling analysis for a scaffold."""
    n_sites: int
    site_spacing_nm: float
    dg_site_repulsion_kj: float     # Per-site penalty from neighbors at full loading
    hill_coefficient: float          # n_Hill: >1 positive, <1 negative, 1 = independent
    cooperativity_type: str          # "positive", "negative", "independent"
    loading_curve: dict              # {fraction: effective_K_factor} at different loadings
    max_practical_loading: float     # Fraction beyond which selectivity degrades
    capacity_mmol_per_g: float       # Total capture capacity
    capacity_mg_per_g: float         # In mg target per g material
    selectivity_degradation_pct: float  # How much selectivity drops at max loading
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# SCAFFOLD SITE PROPERTIES
# Typical site counts and spacings for each scaffold type
# ═══════════════════════════════════════════════════════════════════════════

_SCAFFOLD_SITES = {
    # scaffold_type: (n_sites, spacing_nm, mass_g_per_mol, notes)
    "zeolite_Y":                  (48, 0.74,  15000, "Si/Al ratio dependent"),
    "zeolite_ZSM5":               (28, 0.56,  10000, "Channel intersections"),
    "MOF_UiO66":                  (12, 1.20,   1700, "Zr6 nodes"),
    "MOF_MIL101":                 (18, 2.90,   4500, "Cr3 nodes, large pores"),
    "MIP":                        (1,  0.00,    500, "Single templated cavity"),
    "LDH":                        (24, 0.30,   4000, "Interlayer anion sites"),
    "mesoporous_silica_MCM41":    (15, 3.50,   5000, "Pore surface sites"),
    "COF":                        (8,  1.80,   2000, "Node functional sites"),
    "coordination_cage":          (4,  1.50,   3000, "Interior vertices"),
    "carbon_nanotube":            (10, 1.00,   2000, "Functionalized defects"),
    "dendrimer_PAMAM_G4":         (64, 0.50,  14000, "Surface amine groups"),
    "dna_origami_icosahedron":    (12, 6.00,  12000, "Interior staple termini"),
    "dna_origami_tetrahedron":    (4,  3.50,   3000, "Interior staple termini"),
}


def compute_site_repulsion(target_charge, n_sites, site_spacing_nm,
                            dielectric=15.0, loading_fraction=1.0):
    """Coulombic repulsion between bound ions at neighboring sites.

    At full loading, each site feels repulsion from occupied neighbors.
    ΔG = z²e² / (4πε₀εᵣ × r) per pair, summed over nearest neighbors.

    Args:
        target_charge: Charge of bound ion
        n_sites: Total binding sites
        site_spacing_nm: Distance between adjacent sites
        dielectric: Effective dielectric constant in scaffold
        loading_fraction: What fraction of sites are occupied

    Returns:
        ΔG_repulsion per site in kJ/mol
    """
    if site_spacing_nm <= 0 or n_sites <= 1:
        return 0.0

    # Number of nearest neighbors (approximate)
    if n_sites <= 4:
        n_neighbors = n_sites - 1
    elif n_sites <= 12:
        n_neighbors = min(3, n_sites - 1)
    else:
        n_neighbors = min(6, n_sites - 1)  # Approximate for large arrays

    # Coulomb energy per pair: E = (z1*z2*e²) / (4πε₀εᵣ*r)
    # In practical units: E(kJ/mol) = 1389.4 × z₁×z₂ / (εᵣ × r_nm × 10)
    # where 1389.4 = e²×Nₐ/(4πε₀) in kJ·pm/mol, /10 to convert nm→Å→pm
    z = target_charge
    r_pm = site_spacing_nm * 1000  # nm to pm

    if r_pm <= 0:
        return 0.0

    e_per_pair = 1389.4 * z * z / (dielectric * r_pm)  # kJ/mol

    # Scale by loading: at half loading, average occupancy of neighbors halved
    effective_neighbors = n_neighbors * loading_fraction

    return round(e_per_pair * effective_neighbors, 2)


def compute_hill_cooperativity(n_sites, site_spacing_nm, target_charge,
                                 scaffold_type=""):
    """Estimate Hill coefficient from site geometry.

    n_Hill > 1: positive cooperativity (binding helps next binding)
    n_Hill = 1: independent sites
    n_Hill < 1: negative cooperativity (binding hinders next)

    For metal capture on scaffolds:
    - Close spacing + high charge = negative (repulsion)
    - Allosteric scaffolds = positive (rare in synthetic systems)
    - MIP single-site = n/a (n_Hill = 1)
    """
    if n_sites <= 1:
        return 1.0, "independent"

    # Repulsion-driven negative cooperativity
    # Closer sites + higher charge = more negative
    repulsion_at_full = compute_site_repulsion(target_charge, n_sites,
                                                site_spacing_nm, loading_fraction=1.0)

    if repulsion_at_full > 20:      # Strong repulsion
        n_hill = max(0.3, 1.0 - repulsion_at_full / 200.0)
        coop = "negative"
    elif repulsion_at_full > 5:     # Moderate
        n_hill = max(0.6, 1.0 - repulsion_at_full / 100.0)
        coop = "negative"
    elif repulsion_at_full > 0.5:   # Weak
        n_hill = max(0.85, 1.0 - repulsion_at_full / 50.0)
        coop = "weakly_negative"
    else:
        n_hill = 1.0
        coop = "independent"

    # DNA origami with allosteric lid can show positive cooperativity
    if "dna_origami" in scaffold_type and site_spacing_nm > 4.0:
        n_hill = min(1.5, n_hill + 0.3)
        if n_hill > 1.0:
            coop = "positive"

    return round(n_hill, 2), coop


def compute_loading_curve(target_charge, n_sites, site_spacing_nm, n_points=5):
    """Predict how binding affinity changes with loading fraction.

    Returns dict of {loading_fraction: K_effective / K_intrinsic}.
    """
    curve = {}
    for i in range(n_points + 1):
        frac = i / n_points
        repulsion = compute_site_repulsion(target_charge, n_sites,
                                            site_spacing_nm, loading_fraction=frac)
        # K_eff = K_intrinsic × exp(-ΔG_repulsion / RT)
        R = 8.314e-3
        T = 298.15
        k_factor = math.exp(-repulsion / (R * T)) if repulsion < 100 else 0.0
        curve[round(frac, 2)] = round(k_factor, 4)
    return curve


def compute_capacity(scaffold_type, target_mw_g_mol, target_charge=2,
                      n_sites_override=None, site_spacing_override=None):
    """Predict total capture capacity of a scaffold material.

    Returns capacity in mmol/g and mg/g.
    """
    if scaffold_type in _SCAFFOLD_SITES:
        n_sites, spacing, scaffold_mw, notes = _SCAFFOLD_SITES[scaffold_type]
    else:
        n_sites = 4
        spacing = 1.0
        scaffold_mw = 5000
        notes = "Estimated"

    if n_sites_override:
        n_sites = n_sites_override
    if site_spacing_override:
        spacing = site_spacing_override

    # Loading efficiency: repulsion reduces practical capacity
    hill, _ = compute_hill_cooperativity(n_sites, spacing, target_charge, scaffold_type)
    max_loading = min(1.0, 0.5 + 0.5 * hill)  # Hill < 1 → can't fill all sites

    effective_sites = n_sites * max_loading
    mmol_per_g = (effective_sites / scaffold_mw) * 1000  # mmol per g scaffold
    mg_per_g = mmol_per_g * target_mw_g_mol

    return round(mmol_per_g, 3), round(mg_per_g, 2), round(max_loading, 2)


def analyze_cooperativity(scaffold_type, target_charge=2, target_mw_g_mol=60.0,
                           n_sites_override=None, site_spacing_override=None):
    """Full cooperativity analysis for a scaffold + target combination."""
    if scaffold_type in _SCAFFOLD_SITES:
        n_sites, spacing, scaffold_mw, notes = _SCAFFOLD_SITES[scaffold_type]
    else:
        n_sites = 4
        spacing = 1.0
        scaffold_mw = 5000
        notes = "Estimated"

    if n_sites_override:
        n_sites = n_sites_override
    if site_spacing_override is not None:
        spacing = site_spacing_override

    dg_repulsion = compute_site_repulsion(target_charge, n_sites, spacing)
    hill, coop_type = compute_hill_cooperativity(n_sites, spacing, target_charge, scaffold_type)
    loading = compute_loading_curve(target_charge, n_sites, spacing)
    mmol_g, mg_g, max_load = compute_capacity(scaffold_type, target_mw_g_mol,
                                                target_charge, n_sites_override,
                                                site_spacing_override)

    # Selectivity degradation: at max loading, how much does selectivity drop?
    # Higher repulsion → more degradation for divalent target vs monovalent competitor
    sel_degrad = min(80.0, dg_repulsion * 2.0)  # Rough estimate

    note_parts = []
    if dg_repulsion > 10:
        note_parts.append(f"Significant site-site repulsion ({dg_repulsion:.1f} kJ/mol)")
    if max_load < 0.7:
        note_parts.append(f"Only {max_load*100:.0f}% of sites practically usable")
    if coop_type == "positive":
        note_parts.append("Positive cooperativity — binding improves with loading")

    return CooperativityResult(
        n_sites=n_sites, site_spacing_nm=spacing,
        dg_site_repulsion_kj=dg_repulsion,
        hill_coefficient=hill, cooperativity_type=coop_type,
        loading_curve=loading,
        max_practical_loading=max_load,
        capacity_mmol_per_g=mmol_g, capacity_mg_per_g=mg_g,
        selectivity_degradation_pct=round(sel_degrad, 1),
        notes="; ".join(note_parts),
    )

''')

write_file("core/mass_transport.py", '''\
"""
core/mass_transport.py — Sprint 24: Mass Transport + Diffusion Coupling

Replaces heuristic P_enter with physics-based diffusion modeling.
Stokes-Einstein for bulk, hindered diffusion in pores, Thiele modulus
for diffusion vs reaction limitation, and time-to-capture prediction.

Physics:
  D_bulk = kT / (6πηr_h)                           (Stokes-Einstein)
  D_pore = D_bulk × (1-λ)² × (1 - 2.104λ + ...)   (Renkin hindered)
  φ = L√(k_rxn/D_eff)                              (Thiele modulus)
  t_90 = capacity / (flux × area × 0.9)             (Time to 90% loading)
"""
from dataclasses import dataclass
import math


@dataclass
class DiffusionProfile:
    """Diffusion characterization of a target in a scaffold."""
    d_bulk_m2_s: float              # Stokes-Einstein bulk diffusion
    d_pore_m2_s: float              # Hindered pore diffusion
    hindrance_factor: float         # D_pore / D_bulk (0-1)
    lambda_ratio: float             # r_ion / r_pore (0 = unhindered, >1 = excluded)
    transport_regime: str           # "unhindered", "hindered", "severely_hindered", "excluded"
    thiele_modulus: float           # φ: <1 reaction-limited, >1 diffusion-limited
    effectiveness_factor: float     # η: fraction of scaffold actually used
    rate_limiting_step: str         # "reaction", "pore_diffusion", "external_mass_transfer"
    notes: str = ""

@dataclass
class CaptureKinetics:
    """Time-dependent capture prediction."""
    time_to_50pct_s: float          # Time to 50% loading
    time_to_90pct_s: float          # Time to 90% loading
    time_to_99pct_s: float          # Time to 99% loading
    initial_flux_mol_m2_s: float    # Initial capture rate per unit area
    equilibrium_loading_pct: float  # Predicted final loading
    breakthrough_time_s: float      # For column mode: when target appears in effluent
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

_K_BOLTZMANN = 1.381e-23   # J/K
_VISCOSITY_WATER_25C = 8.9e-4  # Pa·s
_AVOGADRO = 6.022e23


# ═══════════════════════════════════════════════════════════════════════════
# DIFFUSION COEFFICIENTS
# ═══════════════════════════════════════════════════════════════════════════

def stokes_einstein(hydrated_radius_nm, temp_c=25.0, viscosity=None):
    """Bulk diffusion coefficient via Stokes-Einstein.

    D = kT / (6πηr_h)

    Returns D in m²/s. Typical values: 0.5-2 × 10⁻⁹ m²/s for metal ions.
    """
    r_m = hydrated_radius_nm * 1e-9
    T = temp_c + 273.15
    eta = viscosity if viscosity else _VISCOSITY_WATER_25C * (293.15 / T)  # Approx T correction

    D = _K_BOLTZMANN * T / (6.0 * math.pi * eta * r_m)
    return D


def hindered_diffusion(d_bulk, ion_radius_nm, pore_radius_nm):
    """Hindered diffusion in a cylindrical pore (Renkin equation).

    D_pore = D_bulk × (1-λ)² × (1 - 2.104λ + 2.089λ³ - 0.948λ⁵)
    where λ = r_ion / r_pore

    Returns (D_pore, hindrance_factor, lambda_ratio, regime)
    """
    if pore_radius_nm <= 0:
        return d_bulk, 1.0, 0.0, "unhindered"

    lam = ion_radius_nm / pore_radius_nm

    if lam >= 1.0:
        return 0.0, 0.0, lam, "excluded"

    if lam > 0.95:
        return d_bulk * 1e-6, 1e-6, lam, "excluded"

    # Renkin equation
    steric = (1.0 - lam)**2
    hydro = 1.0 - 2.104 * lam + 2.089 * lam**3 - 0.948 * lam**5
    hindrance = steric * hydro
    hindrance = max(0.0, min(1.0, hindrance))

    d_pore = d_bulk * hindrance

    if lam < 0.2:
        regime = "unhindered"
    elif lam < 0.5:
        regime = "hindered"
    elif lam < 0.9:
        regime = "severely_hindered"
    else:
        regime = "excluded"

    return d_pore, hindrance, lam, regime


def compute_thiele_modulus(d_eff, k_rxn_s, particle_radius_m):
    """Thiele modulus: φ = R × √(k/D_eff).

    φ < 0.3: reaction-limited (all scaffold interior is used)
    0.3 < φ < 3: transitional
    φ > 3: diffusion-limited (only outer shell of scaffold is active)

    Returns (φ, effectiveness_factor η)
    """
    if d_eff <= 0 or k_rxn_s <= 0 or particle_radius_m <= 0:
        return 0.0, 1.0

    phi = particle_radius_m * math.sqrt(k_rxn_s / d_eff)

    # Effectiveness factor for sphere: η = (1/φ) × (1/tanh(3φ) - 1/(3φ))
    if phi < 0.01:
        eta = 1.0
    elif phi > 50:
        eta = 1.0 / phi
    else:
        try:
            eta = (1.0 / phi) * (1.0 / math.tanh(3.0 * phi) - 1.0 / (3.0 * phi))
        except (OverflowError, ZeroDivisionError):
            eta = 1.0 / phi

    eta = max(0.001, min(1.0, eta))

    return round(phi, 3), round(eta, 4)


# ═══════════════════════════════════════════════════════════════════════════
# FULL TRANSPORT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def analyze_transport(
    hydrated_radius_nm, pore_diameter_nm,
    particle_diameter_um=10.0,
    k_on_M_s=1e6,
    temp_c=25.0,
    target_charge=2,
):
    """Full mass transport analysis for target in scaffold.

    Args:
        hydrated_radius_nm: Hydrated radius of target ion
        pore_diameter_nm: Scaffold pore diameter
        particle_diameter_um: Scaffold particle size (for Thiele)
        k_on_M_s: Binding on-rate (M⁻¹s⁻¹)
        temp_c: Temperature
        target_charge: For electromigration correction

    Returns:
        DiffusionProfile
    """
    pore_radius_nm = pore_diameter_nm / 2.0

    d_bulk = stokes_einstein(hydrated_radius_nm, temp_c)
    d_pore, hindrance, lam, regime = hindered_diffusion(
        d_bulk, hydrated_radius_nm, pore_radius_nm)

    # Thiele modulus
    particle_r_m = particle_diameter_um * 1e-6 / 2.0
    # Convert k_on (M⁻¹s⁻¹) to pseudo-first-order k (s⁻¹) at 1 µM target
    k_pseudo = k_on_M_s * 1e-6  # s⁻¹ at 1 µM
    d_eff = d_pore if d_pore > 0 else d_bulk * 0.001

    phi, eta = compute_thiele_modulus(d_eff, k_pseudo, particle_r_m)

    # Rate-limiting step
    if regime == "excluded":
        rate_limit = "pore_exclusion"
    elif phi > 3:
        rate_limit = "pore_diffusion"
    elif phi < 0.3:
        rate_limit = "reaction"
    else:
        rate_limit = "mixed"

    notes_parts = []
    if regime == "excluded":
        notes_parts.append(f"Target (r={hydrated_radius_nm:.2f} nm) EXCLUDED from "
                           f"pore (r={pore_radius_nm:.2f} nm)")
    elif regime == "severely_hindered":
        notes_parts.append(f"Severely hindered transport: D_pore/D_bulk = {hindrance:.4f}")
    if phi > 3:
        notes_parts.append(f"Diffusion-limited (φ={phi:.1f}): only {eta*100:.0f}% "
                           f"of scaffold interior is effective")

    return DiffusionProfile(
        d_bulk_m2_s=d_bulk, d_pore_m2_s=d_pore,
        hindrance_factor=round(hindrance, 6),
        lambda_ratio=round(lam, 3),
        transport_regime=regime,
        thiele_modulus=phi, effectiveness_factor=eta,
        rate_limiting_step=rate_limit,
        notes="; ".join(notes_parts),
    )


def predict_capture_time(
    target_conc_uM, capacity_mmol_g, material_g_per_L,
    k_on_M_s=1e6, effectiveness=1.0,
    flow_rate_mL_min=0.0, column_volume_mL=0.0,
):
    """Predict time to reach various loading levels.

    For batch mode: simple kinetic model
    For column mode: breakthrough prediction

    Args:
        target_conc_uM: Target concentration in solution
        capacity_mmol_g: Material capacity
        material_g_per_L: Amount of material per liter
        k_on_M_s: Binding on-rate
        effectiveness: Thiele effectiveness factor
        flow_rate_mL_min: >0 for column mode
        column_volume_mL: Column volume for breakthrough
    """
    # Total capacity in solution
    cap_total_uM = capacity_mmol_g * material_g_per_L * 1000  # µM equivalent

    if cap_total_uM <= 0:
        return CaptureKinetics(0, 0, 0, 0, 0, 0, "No capacity")

    # Effective k_on accounting for transport
    k_eff = k_on_M_s * effectiveness
    conc_M = target_conc_uM * 1e-6

    # Pseudo-first-order rate
    k_pseudo = k_eff * conc_M  # s⁻¹

    if k_pseudo <= 0:
        return CaptureKinetics(1e12, 1e12, 1e12, 0, 0, 0, "Zero rate")

    # Time to reach fraction f: t = -ln(1-f) / k_pseudo (first-order approximation)
    t_50 = -math.log(0.5) / k_pseudo
    t_90 = -math.log(0.1) / k_pseudo
    t_99 = -math.log(0.01) / k_pseudo

    # Initial flux
    flux = k_eff * conc_M**2 * 1000  # mol/m²/s (order of magnitude)

    # Equilibrium loading
    eq_load = min(100.0, (target_conc_uM / cap_total_uM) * 100) if cap_total_uM > 0 else 0

    # Breakthrough for column
    breakthrough = 0.0
    if flow_rate_mL_min > 0 and column_volume_mL > 0:
        bed_volumes = cap_total_uM / max(0.001, target_conc_uM)
        breakthrough = bed_volumes * column_volume_mL / flow_rate_mL_min * 60  # seconds

    notes = ""
    if t_90 > 3600:
        notes = f"Slow: {t_90/3600:.1f} hours to 90% capture"
    elif t_90 < 60:
        notes = f"Fast: {t_90:.0f} seconds to 90% capture"

    return CaptureKinetics(
        time_to_50pct_s=round(t_50, 1),
        time_to_90pct_s=round(t_90, 1),
        time_to_99pct_s=round(t_99, 1),
        initial_flux_mol_m2_s=flux,
        equilibrium_loading_pct=round(eq_load, 1),
        breakthrough_time_s=round(breakthrough, 1),
        notes=notes,
    )

''')

write_file("tests/test_sprint23_24.py", '''\
"""tests/test_sprint23_24.py — Sprints 23+24: System-Level Prediction (25 tests)"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cooperativity import (
    compute_site_repulsion, compute_hill_cooperativity,
    compute_loading_curve, compute_capacity, analyze_cooperativity,
)
from core.mass_transport import (
    stokes_einstein, hindered_diffusion, compute_thiele_modulus,
    analyze_transport, predict_capture_time,
)

# ═══════════════════════════════════════════════════════════════════════════
# SPRINT 23: COOPERATIVITY
# ═══════════════════════════════════════════════════════════════════════════

def test_repulsion_close_sites():
    """Close sites (0.3 nm) with divalent ions should have large repulsion."""
    dg = compute_site_repulsion(2, 24, 0.30)
    assert dg > 5, f"Close divalent sites should repel strongly, got {dg}"
    print(f"  \\u2705 test_repulsion_close: LDH 0.3nm spacing, dG={dg:.1f} kJ/mol")

def test_repulsion_far_sites():
    """Far sites (6 nm, DNA origami) should have negligible repulsion."""
    dg = compute_site_repulsion(2, 12, 6.0)
    assert dg < 2, f"6nm spacing should have minimal repulsion, got {dg}"
    print(f"  \\u2705 test_repulsion_far: DNA origami 6nm, dG={dg:.2f} kJ/mol")

def test_repulsion_monovalent_less():
    """Monovalent ions should repel less than divalent at same spacing."""
    dg1 = compute_site_repulsion(1, 12, 1.0)
    dg2 = compute_site_repulsion(2, 12, 1.0)
    assert dg1 < dg2
    assert dg2 / dg1 > 3  # z² scaling: 4/1
    print(f"  \\u2705 test_repulsion_charge: z=1 dG={dg1:.2f} vs z=2 dG={dg2:.2f} (ratio={dg2/dg1:.1f}x)")

def test_hill_negative_close_sites():
    """Close, highly-charged sites should give negative cooperativity."""
    hill, ctype = compute_hill_cooperativity(48, 0.30, 2)
    assert hill < 1.0, f"Close sites should be negative, got n_Hill={hill}"
    assert "negative" in ctype
    print(f"  \\u2705 test_hill_negative: n_Hill={hill:.2f}, type={ctype}")

def test_hill_independent_mip():
    """MIP single-site should be independent (n_Hill=1)."""
    hill, ctype = compute_hill_cooperativity(1, 0.0, 2)
    assert hill == 1.0
    assert ctype == "independent"
    print(f"  \\u2705 test_hill_mip: n_Hill={hill}, type={ctype}")

def test_hill_positive_dna_origami():
    """DNA origami with wide spacing can show positive cooperativity."""
    hill, ctype = compute_hill_cooperativity(12, 6.0, 2, "dna_origami_icosahedron")
    assert hill >= 1.0, f"DNA origami should have positive/neutral cooperativity"
    print(f"  \\u2705 test_hill_dna_origami: n_Hill={hill:.2f}, type={ctype}")

def test_loading_curve_decreases():
    """K_effective should decrease with loading for repulsive systems."""
    curve = compute_loading_curve(2, 48, 0.30)
    assert curve[0.0] > curve[1.0], \\
        f"K should decrease: empty={curve[0.0]:.4f}, full={curve[1.0]:.4f}"
    print(f"  \\u2705 test_loading_curve: K at 0%={curve[0.0]:.4f}, 50%={curve[0.4]:.4f}, 100%={curve[1.0]:.4f}")

def test_capacity_zeolite():
    """Zeolite should have measurable capacity in mmol/g."""
    mmol, mg, max_load = compute_capacity("zeolite_Y", 58.7)  # Ni2+ MW=58.7
    assert mmol > 0.1
    assert mg > 5
    assert 0 < max_load <= 1.0
    print(f"  \\u2705 test_capacity_zeolite: {mmol:.3f} mmol/g, {mg:.1f} mg Ni/g, max_load={max_load:.0%}")

def test_capacity_mip_single():
    """MIP single-site capacity should be lower than multi-site scaffolds."""
    mmol_mip, _, _ = compute_capacity("MIP", 207.2)  # Pb2+ MW
    mmol_zeo, _, _ = compute_capacity("zeolite_Y", 207.2)
    assert mmol_mip < mmol_zeo, "MIP should have less capacity than zeolite"
    print(f"  \\u2705 test_capacity_mip: MIP={mmol_mip:.3f} vs zeolite={mmol_zeo:.3f} mmol/g")

def test_full_cooperativity_analysis():
    """Full analysis should return all fields."""
    r = analyze_cooperativity("zeolite_Y", target_charge=2, target_mw_g_mol=63.5)
    assert r.n_sites == 48
    assert r.site_spacing_nm == 0.74
    assert r.capacity_mmol_per_g > 0
    assert r.hill_coefficient < 1.0  # Should be negative for zeolite
    assert r.loading_curve is not None
    assert len(r.loading_curve) > 3
    print(f"  \\u2705 test_full_coop: n_Hill={r.hill_coefficient:.2f}, "
          f"cap={r.capacity_mg_per_g:.1f} mg/g, max_load={r.max_practical_loading:.0%}")

def test_dendrimer_many_sites():
    """Dendrimer G4 (64 sites at 0.5 nm) should show strong negative cooperativity."""
    r = analyze_cooperativity("dendrimer_PAMAM_G4", target_charge=3)
    assert r.hill_coefficient < 0.95, f"64 sites at 0.5nm with 3+ should be negative, got {r.hill_coefficient}"
    assert r.cooperativity_type == "negative"
    print(f"  \\u2705 test_dendrimer: n_Hill={r.hill_coefficient:.2f}, "
          f"repulsion={r.dg_site_repulsion_kj:.1f} kJ/mol")

# ═══════════════════════════════════════════════════════════════════════════
# SPRINT 24: MASS TRANSPORT
# ═══════════════════════════════════════════════════════════════════════════

def test_stokes_einstein_reasonable():
    """D for typical metal ion should be ~0.5-2 × 10⁻⁹ m²/s."""
    D = stokes_einstein(0.21)  # Ni2+ hydrated radius
    assert 1e-10 < D < 5e-9, f"D={D:.2e} outside expected range"
    print(f"  \\u2705 test_stokes_einstein: D(Ni2+)={D:.2e} m²/s")

def test_hindered_diffusion_unhindered():
    """Large pore (3.5 nm) should barely hinder small ion (0.2 nm)."""
    _, hindrance, lam, regime = hindered_diffusion(1e-9, 0.2, 1.75)
    assert hindrance > 0.4
    assert regime == "unhindered"
    print(f"  \\u2705 test_hind_unhindered: λ={lam:.3f}, H={hindrance:.3f}, regime={regime}")

def test_hindered_diffusion_severe():
    """ZSM-5 (0.28 nm radius) should severely hinder Pb2+ (0.24 nm hydrated)."""
    _, hindrance, lam, regime = hindered_diffusion(1e-9, 0.24, 0.28)
    assert hindrance < 0.1
    assert regime in ("severely_hindered", "excluded")
    print(f"  \\u2705 test_hind_severe: λ={lam:.3f}, H={hindrance:.4f}, regime={regime}")

def test_hindered_diffusion_excluded():
    """Ion larger than pore should be excluded."""
    _, hindrance, lam, regime = hindered_diffusion(1e-9, 0.30, 0.25)
    assert hindrance < 0.001
    assert regime == "excluded"
    print(f"  \\u2705 test_hind_excluded: λ={lam:.3f}, regime={regime}")

def test_thiele_reaction_limited():
    """Small particle + slow reaction → reaction-limited."""
    phi, eta = compute_thiele_modulus(1e-9, 1.0, 1e-6)
    assert phi < 0.3, f"Should be reaction-limited, φ={phi}"
    assert eta > 0.9
    print(f"  \\u2705 test_thiele_rxn: φ={phi:.3f}, η={eta:.4f}")

def test_thiele_diffusion_limited():
    """Large particle + fast reaction → diffusion-limited."""
    phi, eta = compute_thiele_modulus(1e-12, 100.0, 1e-3)
    assert phi > 3, f"Should be diffusion-limited, φ={phi}"
    assert eta < 0.5
    print(f"  \\u2705 test_thiele_diff: φ={phi:.1f}, η={eta:.4f}")

def test_transport_analysis_zeolite():
    """Full transport analysis for Pb2+ in zeolite Y."""
    r = analyze_transport(hydrated_radius_nm=0.24, pore_diameter_nm=0.74,
                           particle_diameter_um=5.0, k_on_M_s=1e6)
    assert r.d_bulk_m2_s > 0
    assert r.hindrance_factor < 1.0
    assert r.lambda_ratio > 0
    print(f"  \\u2705 test_transport_zeolite: D_bulk={r.d_bulk_m2_s:.2e}, "
          f"D_pore={r.d_pore_m2_s:.2e}, λ={r.lambda_ratio:.3f}, "
          f"regime={r.transport_regime}")

def test_transport_dna_origami():
    """DNA origami (8 nm pore) should be unhindered for any ion."""
    r = analyze_transport(hydrated_radius_nm=0.30, pore_diameter_nm=8.0)
    assert r.transport_regime == "unhindered"
    assert r.hindrance_factor > 0.5
    print(f"  \\u2705 test_transport_dna: regime={r.transport_regime}, H={r.hindrance_factor:.4f}")

def test_capture_time_fast():
    """High k_on + low concentration should give short capture time."""
    ct = predict_capture_time(target_conc_uM=10.0, capacity_mmol_g=1.0,
                               material_g_per_L=1.0, k_on_M_s=1e6)
    assert ct.time_to_90pct_s > 0
    assert ct.time_to_90pct_s < ct.time_to_99pct_s
    print(f"  \\u2705 test_capture_fast: t50={ct.time_to_50pct_s:.0f}s, "
          f"t90={ct.time_to_90pct_s:.0f}s, t99={ct.time_to_99pct_s:.0f}s")

def test_capture_time_column():
    """Column mode should predict breakthrough time."""
    ct = predict_capture_time(target_conc_uM=1.0, capacity_mmol_g=2.0,
                               material_g_per_L=10.0,
                               flow_rate_mL_min=1.0, column_volume_mL=5.0)
    assert ct.breakthrough_time_s > 0
    print(f"  \\u2705 test_capture_column: breakthrough={ct.breakthrough_time_s:.0f}s "
          f"({ct.breakthrough_time_s/60:.1f} min)")

def test_effectiveness_reduces_capture():
    """Low effectiveness factor should slow capture."""
    ct_full = predict_capture_time(1.0, 1.0, 1.0, 1e6, effectiveness=1.0)
    ct_low = predict_capture_time(1.0, 1.0, 1.0, 1e6, effectiveness=0.1)
    assert ct_low.time_to_90pct_s > ct_full.time_to_90pct_s
    print(f"  \\u2705 test_effectiveness: η=1.0 t90={ct_full.time_to_90pct_s:.0f}s, "
          f"η=0.1 t90={ct_low.time_to_90pct_s:.0f}s")

if __name__ == "__main__":
    print("\\n\\U0001f9ea Sprints 23+24 \\u2014 System-Level Prediction\\n")
    print("Sprint 23 — Cooperativity:")
    test_repulsion_close_sites(); test_repulsion_far_sites()
    test_repulsion_monovalent_less(); test_hill_negative_close_sites()
    test_hill_independent_mip(); test_hill_positive_dna_origami()
    test_loading_curve_decreases(); test_capacity_zeolite()
    test_capacity_mip_single(); test_full_cooperativity_analysis()
    test_dendrimer_many_sites()
    print("\\nSprint 24 — Mass Transport:")
    test_stokes_einstein_reasonable(); test_hindered_diffusion_unhindered()
    test_hindered_diffusion_severe(); test_hindered_diffusion_excluded()
    test_thiele_reaction_limited(); test_thiele_diffusion_limited()
    test_transport_analysis_zeolite(); test_transport_dna_origami()
    test_capture_time_fast(); test_capture_time_column()
    test_effectiveness_reduces_capture()
    print("\\n\\u2705 All Sprint 23+24 tests passed! (22/22)")
    print("\\n\\U0001f389 SYSTEM-LEVEL PREDICTION OPERATIONAL\\n")

''')


print("""
\u2705 Sprint 23+24 files created!

Sprint 23 — Cooperativity (236 lines):
  Site-site Coulombic repulsion (confined dielectric \u03b5=15)
  Hill coefficient from site geometry (positive/negative/independent)
  Loading curves: K_eff vs loading fraction
  Capacity prediction: mmol/g and mg/g for 13 scaffold types

Sprint 24 — Mass Transport (275 lines):
  Stokes-Einstein bulk diffusion
  Renkin hindered pore diffusion
  Thiele modulus + effectiveness factor
  Capture time prediction (batch + column breakthrough)

Run: python tests/test_sprint23_24.py
""")