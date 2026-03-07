"""
core/inorganic_surfaces.py — Functionalized Silica & Mineral Surface Adapter

Non-carbon binding systems: surface complexation on inorganic materials.

The binding pocket is the surface itself — hydroxyl groups, lattice oxygens,
grafted functional groups, and pore geometry create the interaction site.

Physics:
  Surface complexation model (Stumm & Morgan):
    ≡S-OH + M²⁺ → ≡S-O-M⁺ + H⁺          (inner-sphere)
    ≡S-OH + M²⁺ → ≡S-OH···M²⁺             (outer-sphere)

  log K_ads = log K_int + z × F × ψ₀/(RT ln10)
    where K_int = intrinsic surface complexation constant
    ψ₀ = surface potential (from surface charge model)
    z = charge change upon binding

  For grafted surfaces (e.g. APTES-SiO₂):
    The organic functional group IS the donor, but the scaffold is inorganic.
    Binding energy from ideal_pocket donor model.
    Capacity from site density × surface area.

Material systems:
  SILICA (native):          Si-OH sites, weak cation exchange
  SILICA (grafted):         Si-O-Si-(CH₂)₃-NH₂ (APTES) or -SH (MPTMS)
  MESOPOROUS SILICA:        MCM-41 (2-4nm pores), SBA-15 (5-15nm pores)
  IRON OXIDE:               Fe-OH sites, strong for As, Pb, Cr
  MAGNETITE:                Fe₃O₄, magnetically separable
  ALUMINA:                  Al-OH, amphoteric, strong for F⁻, PO₄³⁻
  TITANIA:                  Ti-OH, photocatalytic + adsorption
  HYDROXYAPATITE:           Ca₅(PO₄)₃OH, strong for Pb, F, Cd
  MONTMORILLONITE:          Interlayer cation exchange, CEC 80-120 meq/100g
  ZEOLITE (generic):        Framework cation exchange (detailed in future adapter)

References:
  Stumm & Morgan, Aquatic Chemistry, 3rd ed. (1996) — surface complexation
  Kosmulski, Surface Charging and PZC, 2nd ed. (2009) — PZC database
  Sposito, The Surface Chemistry of Soils (1984) — clay CEC
  Mohan & Pittman, J. Hazard. Mater. 142:1 (2007) — arsenic on iron oxides
  Babel & Kurniawan, J. Hazard. Mater. 97:219 (2003) — heavy metal adsorbents
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# SURFACE MATERIAL DATABASE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SurfaceMaterial:
    """Properties of an inorganic surface for binding."""
    name: str
    formula: str
    surface_type: str            # "oxide", "clay", "phosphate", "silicate"

    # Surface chemistry
    surface_site: str            # dominant site, e.g. "Si-OH", "Fe-OH"
    site_density_per_nm2: float  # reactive sites per nm²
    pKa1: float                  # ≡S-OH₂⁺ → ≡S-OH + H⁺
    pKa2: float                  # ≡S-OH → ≡S-O⁻ + H⁺
    pzc: float                   # point of zero charge

    # Physical
    surface_area_m2_g: float     # BET specific surface area (m²/g)
    pore_diameter_nm: float      # 0 for non-porous, >0 for mesoporous
    density_g_cm3: float         # bulk density

    # Capabilities
    donor_type_native: str       # HSAB class of native surface: "hard", "borderline"
    magnetic: bool = False       # magnetically separable?
    photocatalytic: bool = False

    # Economics
    cost_usd_per_kg: float = 0.0
    commercial_availability: str = "commodity"  # "commodity", "specialty", "custom"

    # Binding
    max_capacity_mg_g: float = 0.0  # typical max capacity for divalent metals (mg/g)
    pH_optimal_low: float = 2.0
    pH_optimal_high: float = 12.0
    notes: str = ""


SURFACE_MATERIALS = {
    # ── SILICA ──
    "SiO2": SurfaceMaterial(
        "amorphous silica", "SiO2", "oxide",
        "Si-OH", 4.6, 1.0, 6.8, 3.9,
        surface_area_m2_g=200, pore_diameter_nm=0, density_g_cm3=2.2,
        donor_type_native="hard", cost_usd_per_kg=5,
        max_capacity_mg_g=15, pH_optimal_low=4, pH_optimal_high=9,
        notes="Weak native binding; mainly used as scaffold for grafting"),

    "SiO2_APTES": SurfaceMaterial(
        "APTES-grafted silica", "SiO2-NH2", "oxide",
        "R-NH2", 2.5, 8.0, 10.5, 7.5,
        surface_area_m2_g=180, pore_diameter_nm=0, density_g_cm3=2.1,
        donor_type_native="borderline", cost_usd_per_kg=25,
        max_capacity_mg_g=80, pH_optimal_low=3, pH_optimal_high=8,
        notes="Amine-functionalized; chelates Cu, Ni, Zn, Pb"),

    "SiO2_MPTMS": SurfaceMaterial(
        "thiol-grafted silica", "SiO2-SH", "oxide",
        "R-SH", 2.0, 8.5, 11.0, 6.0,
        surface_area_m2_g=170, pore_diameter_nm=0, density_g_cm3=2.1,
        donor_type_native="soft", cost_usd_per_kg=35,
        max_capacity_mg_g=120, pH_optimal_low=2, pH_optimal_high=7,
        notes="Thiol-functionalized; selective for Hg, Ag, Au, Pb"),

    "SiO2_DTPA": SurfaceMaterial(
        "DTPA-grafted silica", "SiO2-DTPA", "oxide",
        "R-DTPA", 1.5, 2.0, 10.0, 5.5,
        surface_area_m2_g=150, pore_diameter_nm=0, density_g_cm3=2.1,
        donor_type_native="hard", cost_usd_per_kg=80,
        max_capacity_mg_g=150, pH_optimal_low=3, pH_optimal_high=8,
        notes="Polyaminocarboxylate; broadband chelation, high capacity"),

    "SiO2_phosphonate": SurfaceMaterial(
        "phosphonate-grafted silica", "SiO2-PO3H2", "oxide",
        "R-PO3H2", 2.0, 1.5, 7.0, 4.0,
        surface_area_m2_g=160, pore_diameter_nm=0, density_g_cm3=2.1,
        donor_type_native="hard", cost_usd_per_kg=50,
        max_capacity_mg_g=90, pH_optimal_low=2, pH_optimal_high=7,
        notes="Strong for lanthanides, actinides, Fe³⁺, Al³⁺"),

    # ── MESOPOROUS SILICA ──
    "MCM-41": SurfaceMaterial(
        "MCM-41", "SiO2", "oxide",
        "Si-OH", 4.0, 1.0, 6.8, 3.5,
        surface_area_m2_g=1000, pore_diameter_nm=3.0, density_g_cm3=1.8,
        donor_type_native="hard", cost_usd_per_kg=200,
        max_capacity_mg_g=50, pH_optimal_low=3, pH_optimal_high=9,
        notes="Ordered mesoporous; 2-4 nm pores; high SA for grafting"),

    "SBA-15": SurfaceMaterial(
        "SBA-15", "SiO2", "oxide",
        "Si-OH", 3.5, 1.0, 6.8, 3.5,
        surface_area_m2_g=800, pore_diameter_nm=8.0, density_g_cm3=1.7,
        donor_type_native="hard", cost_usd_per_kg=250,
        max_capacity_mg_g=60, pH_optimal_low=3, pH_optimal_high=9,
        notes="Large-pore mesoporous; 5-15 nm; better for large molecule access"),

    # ── IRON OXIDES ──
    "Fe2O3": SurfaceMaterial(
        "hematite", "α-Fe₂O₃", "oxide",
        "Fe-OH", 5.0, 5.0, 10.5, 8.5,
        surface_area_m2_g=30, pore_diameter_nm=0, density_g_cm3=5.3,
        donor_type_native="hard", cost_usd_per_kg=2,
        max_capacity_mg_g=25, pH_optimal_low=4, pH_optimal_high=9,
        notes="Strong for As(V), As(III), Pb²⁺, Cr(VI)"),

    "Fe3O4": SurfaceMaterial(
        "magnetite", "Fe₃O₄", "oxide",
        "Fe-OH", 4.5, 4.5, 10.0, 6.5,
        surface_area_m2_g=50, pore_diameter_nm=0, density_g_cm3=5.2,
        donor_type_native="hard", magnetic=True, cost_usd_per_kg=10,
        max_capacity_mg_g=40, pH_optimal_low=3, pH_optimal_high=9,
        notes="Magnetically separable; easy recovery; good for As, Pb, Cr"),

    "FeOOH": SurfaceMaterial(
        "goethite", "α-FeOOH", "oxide",
        "Fe-OH", 6.0, 5.5, 11.0, 7.5,
        surface_area_m2_g=50, pore_diameter_nm=0, density_g_cm3=4.3,
        donor_type_native="hard", cost_usd_per_kg=5,
        max_capacity_mg_g=60, pH_optimal_low=4, pH_optimal_high=9,
        notes="Highest Fe-OH site density; benchmark for arsenic removal"),

    # ── ALUMINA ──
    "Al2O3": SurfaceMaterial(
        "activated alumina", "γ-Al₂O₃", "oxide",
        "Al-OH", 8.0, 5.0, 11.0, 8.1,
        surface_area_m2_g=200, pore_diameter_nm=5.0, density_g_cm3=3.6,
        donor_type_native="hard", cost_usd_per_kg=3,
        max_capacity_mg_g=30, pH_optimal_low=5, pH_optimal_high=8,
        notes="Strong for F⁻, PO₄³⁻, As(V); amphoteric surface"),

    # ── TITANIA ──
    "TiO2": SurfaceMaterial(
        "anatase titania", "TiO₂", "oxide",
        "Ti-OH", 5.0, 3.5, 8.0, 5.8,
        surface_area_m2_g=50, pore_diameter_nm=0, density_g_cm3=3.9,
        donor_type_native="hard", photocatalytic=True, cost_usd_per_kg=15,
        max_capacity_mg_g=20, pH_optimal_low=3, pH_optimal_high=8,
        notes="Adsorption + photocatalytic degradation; dual-mode"),

    # ── PHOSPHATE MINERALS ──
    "hydroxyapatite": SurfaceMaterial(
        "hydroxyapatite", "Ca₅(PO₄)₃OH", "phosphate",
        "≡P-OH / Ca²⁺", 3.0, 5.5, 8.5, 7.0,
        surface_area_m2_g=60, pore_diameter_nm=0, density_g_cm3=3.2,
        donor_type_native="hard", cost_usd_per_kg=10,
        max_capacity_mg_g=200, pH_optimal_low=4, pH_optimal_high=9,
        notes="Exceptional for Pb²⁺ (forms pyromorphite); also F⁻, Cd²⁺"),

    # ── CLAYS ──
    "montmorillonite": SurfaceMaterial(
        "montmorillonite", "Na-MMT", "clay",
        "interlayer", 0.0, 0.0, 0.0, 0.0,  # CEC, not surface complexation
        surface_area_m2_g=750, pore_diameter_nm=1.2, density_g_cm3=2.4,
        donor_type_native="hard", cost_usd_per_kg=1,
        max_capacity_mg_g=40, pH_optimal_low=3, pH_optimal_high=10,
        notes="CEC ~100 meq/100g; interlayer cation exchange; very cheap"),

    "kaolinite": SurfaceMaterial(
        "kaolinite", "Al₂Si₂O₅(OH)₄", "clay",
        "Al-OH / Si-OH", 3.0, 3.0, 7.5, 4.5,
        surface_area_m2_g=15, pore_diameter_nm=0, density_g_cm3=2.6,
        donor_type_native="hard", cost_usd_per_kg=0.5,
        max_capacity_mg_g=10, pH_optimal_low=4, pH_optimal_high=8,
        notes="Low capacity but ultra-cheap; CEC ~10 meq/100g"),

    # ── ZEOLITES ──
    "zeolite_NaA": SurfaceMaterial(
        "zeolite NaA (LTA)", "Na₁₂Al₁₂Si₁₂O₄₈", "silicate",
        "framework O + Na⁺", 0.0, 0.0, 0.0, 0.0,
        surface_area_m2_g=500, pore_diameter_nm=0.41, density_g_cm3=2.0,
        donor_type_native="hard", cost_usd_per_kg=3,
        max_capacity_mg_g=100, pH_optimal_low=4, pH_optimal_high=10,
        notes="4Å pore; size-selective cation exchange; NH₄⁺/K⁺/Cs⁺ selective"),

    "zeolite_NaX": SurfaceMaterial(
        "zeolite NaX (FAU)", "Na₈₆Al₈₆Si₁₀₆O₃₈₄", "silicate",
        "framework O + Na⁺", 0.0, 0.0, 0.0, 0.0,
        surface_area_m2_g=700, pore_diameter_nm=0.74, density_g_cm3=1.9,
        donor_type_native="hard", cost_usd_per_kg=5,
        max_capacity_mg_g=150, pH_optimal_low=4, pH_optimal_high=10,
        notes="7.4Å pore; higher capacity; less size-selective than NaA"),

    "clinoptilolite": SurfaceMaterial(
        "clinoptilolite", "(Na,K,Ca)₂₋₃Al₃(Al,Si)₂Si₁₃O₃₆·12H₂O", "silicate",
        "framework O + cation", 0.0, 0.0, 0.0, 0.0,
        surface_area_m2_g=30, pore_diameter_nm=0.40, density_g_cm3=2.2,
        donor_type_native="hard", cost_usd_per_kg=0.5,
        max_capacity_mg_g=50, pH_optimal_low=3, pH_optimal_high=10,
        notes="Natural zeolite; Cs⁺/NH₄⁺ selective; extremely cheap"),
}


# ═══════════════════════════════════════════════════════════════════════════
# SURFACE COMPLEXATION ENERGETICS
# ═══════════════════════════════════════════════════════════════════════════

# Intrinsic surface complexation log K values (inner-sphere)
# ≡S-OH + M²⁺ → ≡S-OM⁺ + H⁺
# Selected values from Dzombak & Morel (1990) for HFO, extended to other surfaces
# Positive log K = favorable binding at the surface pKa

_SURFACE_LOG_K = {
    # (surface_type, metal) → log K_int
    # Iron oxides
    ("Fe-OH", "Pb2+"): 4.7,
    ("Fe-OH", "Cu2+"): 2.9,
    ("Fe-OH", "Zn2+"): 1.0,
    ("Fe-OH", "Cd2+"): 0.5,
    ("Fe-OH", "Ni2+"): 0.4,
    ("Fe-OH", "Hg2+"): 7.8,
    ("Fe-OH", "Fe3+"): 6.0,
    ("Fe-OH", "Cr3+"): 5.5,

    # Alumina
    ("Al-OH", "Pb2+"): 3.5,
    ("Al-OH", "Cu2+"): 2.2,
    ("Al-OH", "Zn2+"): 0.8,
    ("Al-OH", "Cd2+"): 0.3,
    ("Al-OH", "Fe3+"): 5.0,

    # Titania
    ("Ti-OH", "Pb2+"): 3.0,
    ("Ti-OH", "Cu2+"): 2.5,
    ("Ti-OH", "Zn2+"): 0.9,
    ("Ti-OH", "Cd2+"): 0.4,

    # Silica (native — weak)
    ("Si-OH", "Pb2+"): 0.5,
    ("Si-OH", "Cu2+"): -0.5,
    ("Si-OH", "Zn2+"): -1.5,
    ("Si-OH", "Cd2+"): -1.8,

    # Grafted amine (APTES)
    ("R-NH2", "Cu2+"): 5.5,
    ("R-NH2", "Ni2+"): 4.0,
    ("R-NH2", "Zn2+"): 3.5,
    ("R-NH2", "Pb2+"): 3.0,
    ("R-NH2", "Cd2+"): 2.5,
    ("R-NH2", "Fe3+"): 4.5,
    ("R-NH2", "Hg2+"): 4.0,

    # Grafted thiol (MPTMS)
    ("R-SH", "Hg2+"): 12.0,
    ("R-SH", "Ag+"): 10.0,
    ("R-SH", "Au3+"): 11.0,
    ("R-SH", "Pb2+"): 6.0,
    ("R-SH", "Cu2+"): 5.0,
    ("R-SH", "Cd2+"): 4.5,
    ("R-SH", "Zn2+"): 2.0,

    # Grafted DTPA
    ("R-DTPA", "Cu2+"): 8.0,
    ("R-DTPA", "Pb2+"): 7.0,
    ("R-DTPA", "Zn2+"): 6.5,
    ("R-DTPA", "Ni2+"): 7.5,
    ("R-DTPA", "Cd2+"): 6.0,
    ("R-DTPA", "Fe3+"): 9.0,
    ("R-DTPA", "Hg2+"): 8.5,

    # Grafted phosphonate
    ("R-PO3H2", "Fe3+"): 8.0,
    ("R-PO3H2", "Al3+"): 7.5,
    ("R-PO3H2", "La3+"): 7.0,
    ("R-PO3H2", "UO2_2+"): 8.5,
    ("R-PO3H2", "Pb2+"): 4.0,
    ("R-PO3H2", "Cu2+"): 3.5,

    # Hydroxyapatite
    ("≡P-OH / Ca²⁺", "Pb2+"): 8.5,
    ("≡P-OH / Ca²⁺", "Cd2+"): 4.0,
    ("≡P-OH / Ca²⁺", "Zn2+"): 3.0,
    ("≡P-OH / Ca²⁺", "Cu2+"): 3.5,
}

# Cation exchange selectivity for clays/zeolites (Eisenman series)
# Relative to Na⁺ = 1.0
_CEC_SELECTIVITY = {
    "montmorillonite": {
        "Cs+": 8.0, "Rb+": 5.0, "K+": 3.5, "Na+": 1.0, "Li+": 0.8,
        "Ba2+": 5.0, "Pb2+": 4.5, "Cu2+": 2.5, "Zn2+": 2.0,
        "Ca2+": 1.8, "Mg2+": 1.5, "Cd2+": 2.2, "Ni2+": 2.0,
    },
    "clinoptilolite": {
        "Cs+": 25.0, "Rb+": 12.0, "K+": 5.0, "NH4+": 8.0,
        "Na+": 1.0, "Li+": 0.5,
        "Ba2+": 3.0, "Pb2+": 6.0, "Ca2+": 1.5, "Mg2+": 0.8,
    },
    "zeolite_NaA": {
        "Cs+": 0.5, "K+": 5.0, "Na+": 1.0, "Li+": 1.5,  # 4Å pore excludes Cs
        "Ca2+": 3.0, "Mg2+": 2.0, "Zn2+": 2.5,
    },
    "zeolite_NaX": {
        "Cs+": 15.0, "K+": 4.0, "Na+": 1.0,
        "Ba2+": 6.0, "Pb2+": 8.0, "Ca2+": 2.0, "Mg2+": 1.2,
        "Cu2+": 3.0, "Zn2+": 2.5, "Cd2+": 3.5,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# BINDING PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SurfaceBindingResult:
    """Predicted binding of a metal on a surface."""
    surface_name: str
    target: str
    mechanism: str              # "inner_sphere", "outer_sphere", "ion_exchange", "grafted_chelation"
    log_K_surface: float        # effective log K at the surface
    pH_dependent: bool = True
    pH_optimal: float = 7.0
    capacity_mg_g: float = 0.0
    capacity_mmol_g: float = 0.0
    selectivity_vs: dict = field(default_factory=dict)  # {competitor: selectivity_factor}
    kinetics: str = "fast"      # "fast", "moderate", "slow"
    regenerable: bool = True
    magnetic_separation: bool = False
    cost_per_kg: float = 0.0
    notes: str = ""


def predict_surface_binding(
    target: str,
    surface_key: str,
    pH: float = 7.0,
    competitors: list = None,
) -> SurfaceBindingResult:
    """Predict binding of a target ion on a surface material.

    Args:
        target:      Metal ion, e.g. "Pb2+"
        surface_key: Key in SURFACE_MATERIALS, e.g. "SiO2_MPTMS"
        pH:          Working pH
        competitors: List of competitor ions for selectivity

    Returns:
        SurfaceBindingResult with predicted log K and capacity
    """
    if surface_key not in SURFACE_MATERIALS:
        raise ValueError(f"Unknown surface: {surface_key}. "
                         f"Options: {sorted(SURFACE_MATERIALS)}")

    mat = SURFACE_MATERIALS[surface_key]

    # Determine mechanism
    if mat.surface_type == "clay" or mat.surface_type == "silicate":
        mechanism = "ion_exchange"
    elif "R-" in mat.surface_site:
        mechanism = "grafted_chelation"
    else:
        mechanism = "inner_sphere"

    # Get base log K
    if mechanism == "ion_exchange":
        log_K = _ion_exchange_log_K(target, surface_key)
    else:
        log_K = _surface_log_K(mat.surface_site, target)

    # pH correction for surface complexation
    if mechanism in ("inner_sphere", "grafted_chelation") and mat.pKa2 > 0:
        # Binding is optimal near pKa2 (deprotonated surface)
        # Reduced below pKa2 (protonated surface competes)
        # Reduced at very high pH (metal hydroxide precipitation)
        if pH < mat.pKa2 - 2:
            # Surface mostly protonated — reduced binding
            log_K -= (mat.pKa2 - 2 - pH) * 0.5
        elif pH > 10:
            # Metal may precipitate as hydroxide
            log_K -= (pH - 10) * 0.3

    # Capacity
    target_charge = _parse_charge(target)
    target_mw = _approx_mw(target)
    if mechanism == "ion_exchange":
        # CEC-based capacity
        cec_meq = 100  # rough default meq/100g
        capacity_mmol_g = cec_meq / (100 * max(target_charge, 1))
    else:
        # Site-density-based capacity
        sites_per_nm2 = mat.site_density_per_nm2
        sa = mat.surface_area_m2_g
        # sites/g = sites/nm² × nm²/m² × m²/g
        sites_per_g = sites_per_nm2 * 1e18 * sa
        capacity_mmol_g = sites_per_g / 6.022e20  # Avogadro
    capacity_mg_g = capacity_mmol_g * target_mw

    # Clamp to literature max
    capacity_mg_g = min(capacity_mg_g, mat.max_capacity_mg_g * 1.5)

    # Selectivity vs competitors
    selectivity = {}
    if competitors:
        for comp in competitors:
            if mechanism == "ion_exchange":
                comp_log_K = _ion_exchange_log_K(comp, surface_key)
            else:
                comp_log_K = _surface_log_K(mat.surface_site, comp)
            selectivity[comp] = round(log_K - comp_log_K, 1)

    # Kinetics
    if mechanism == "ion_exchange":
        kinetics = "fast"  # minutes
    elif mechanism == "grafted_chelation":
        kinetics = "moderate"  # 10-60 min
    else:
        kinetics = "moderate" if mat.surface_area_m2_g > 100 else "slow"

    return SurfaceBindingResult(
        surface_name=mat.name,
        target=target,
        mechanism=mechanism,
        log_K_surface=round(log_K, 1),
        pH_optimal=round((mat.pH_optimal_low + mat.pH_optimal_high) / 2, 1),
        capacity_mg_g=round(capacity_mg_g, 1),
        capacity_mmol_g=round(capacity_mmol_g, 3),
        selectivity_vs=selectivity,
        kinetics=kinetics,
        regenerable=True,
        magnetic_separation=mat.magnetic,
        cost_per_kg=mat.cost_usd_per_kg,
        notes=mat.notes,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SURFACE RECOMMENDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def recommend_surface(
    target: str,
    pH: float = 7.0,
    competitors: list = None,
    require_magnetic: bool = False,
    max_cost_per_kg: float = None,
    min_capacity_mg_g: float = 0.0,
) -> list:
    """Rank all surfaces for a given target.

    Returns list of SurfaceBindingResult, sorted by log K (best first).
    """
    results = []
    for key, mat in SURFACE_MATERIALS.items():
        if require_magnetic and not mat.magnetic:
            continue
        if max_cost_per_kg and mat.cost_usd_per_kg > max_cost_per_kg:
            continue

        try:
            r = predict_surface_binding(target, key, pH, competitors)
            if r.capacity_mg_g >= min_capacity_mg_g:
                results.append(r)
        except Exception:
            continue

    results.sort(key=lambda r: r.log_K_surface, reverse=True)
    return results


def compare_surfaces_for_target(target, pH=7.0, competitors=None):
    """Pretty-print all surfaces ranked for a target."""
    results = recommend_surface(target, pH, competitors)
    print()
    print(f"  MABE Surface Recommendation — {target} at pH {pH}")
    print(f"  {'Surface':25s}  {'Mechanism':20s}  {'logK':>5s}  "
          f"{'Cap(mg/g)':>9s}  {'$/kg':>5s}  Notes")
    print(f"  {'─'*100}")
    for r in results:
        sel_str = ""
        if r.selectivity_vs:
            sel_str = " | ".join(f"vs {k}: {v:+.1f}" for k, v in r.selectivity_vs.items())
        print(f"  {r.surface_name:25s}  {r.mechanism:20s}  "
              f"{r.log_K_surface:5.1f}  {r.capacity_mg_g:9.1f}  "
              f"{r.cost_per_kg:5.0f}  {sel_str or r.notes[:40]}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _surface_log_K(site_type, metal):
    """Look up intrinsic surface complexation log K."""
    return _SURFACE_LOG_K.get((site_type, metal), -2.0)


def _ion_exchange_log_K(metal, surface_key):
    """Compute effective log K from CEC selectivity coefficients."""
    # Map surface_key to CEC table
    cec_table = None
    for clay_name, table in _CEC_SELECTIVITY.items():
        if clay_name in surface_key:
            cec_table = table
            break

    if cec_table is None:
        return 0.0  # no data

    sel = cec_table.get(metal, 1.0)
    # Convert selectivity coefficient to approximate log K
    # log K ≈ log(selectivity) + baseline (~1.0 for Na+)
    return round(math.log10(max(sel, 0.01)) + 1.0, 1)


def _parse_charge(target):
    if "3+" in target: return 3
    if "2+" in target: return 2
    if "+" in target: return 1
    return 0


def _approx_mw(target):
    """Approximate atomic weight for capacity calculation."""
    mws = {
        "Pb2+": 207.2, "Cu2+": 63.5, "Zn2+": 65.4, "Ni2+": 58.7,
        "Cd2+": 112.4, "Hg2+": 200.6, "Fe3+": 55.8, "Cr3+": 52.0,
        "Al3+": 27.0, "Mn2+": 54.9, "Co2+": 58.9, "Ca2+": 40.1,
        "Mg2+": 24.3, "Na+": 23.0, "K+": 39.1, "Cs+": 132.9,
        "Ba2+": 137.3, "Sr2+": 87.6, "Ag+": 107.9, "Au3+": 197.0,
        "La3+": 138.9, "UO2_2+": 270.0,
    }
    return mws.get(target, 100.0)


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("MABE Inorganic Surface Adapter — Self-Test")
    print("=" * 70)

    # Pb²⁺ removal — all surfaces
    compare_surfaces_for_target("Pb2+", pH=6.0, competitors=["Ca2+", "Zn2+"])

    # Hg²⁺ — thiol-functionalized should dominate
    compare_surfaces_for_target("Hg2+", pH=5.0, competitors=["Cu2+"])

    # Cs⁺ — zeolites should dominate
    compare_surfaces_for_target("Cs+", pH=7.0, competitors=["Na+", "K+"])

    # Fe³⁺ — phosphonate should be strong
    compare_surfaces_for_target("Fe3+", pH=4.0)
