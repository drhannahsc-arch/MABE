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

