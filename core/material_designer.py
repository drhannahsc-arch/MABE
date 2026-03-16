"""
material_designer.py — Unified Material Design Engine

Designs materials across 4 classes from physics equations and tiered data:
  M1: Porous Frameworks (MOFs, COFs, zeolites)
  M2: Polymeric Sorbents (ion exchange, hydrogels) — next session
  M3: Composite Materials (scaffold-on-support) — next session
  M4: Structural Color (bridges to optical pipeline) — next session
  M5: Unified Designer (cross-material ranking)

Data tiers:
  T1 = Known: published constants, NIST-level, machine-verifiable
  T2 = Solid: well-established equations, textbook physics
  T3 = Conceptual: estimated from analogy, scaling laws, trends

All equations from first principles. No fitting to application-specific data.

References:
  Langmuir I. JACS 1918, 40, 1361.
  Brunauer S, Emmett PH, Teller E. JACS 1938, 60, 309 (BET).
  Do DD. Adsorption Analysis: Equilibria and Kinetics. Imperial College 1998.
  Rouquerol F et al. Adsorption by Powders and Porous Solids. Academic 2014.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from abc import ABC, abstractmethod


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

R_GAS = 8.314          # J/(mol·K)
R_kJ = 8.314e-3        # kJ/(mol·K)
T_STD = 298.15         # K
RT_STD = R_kJ * T_STD  # 2.479 kJ/mol
N_AVO = 6.022e23       # mol⁻¹
PI = math.pi


class DataTier(Enum):
    """Data quality tier for parameter provenance tracking."""
    T1_KNOWN = 1       # Published constants, machine-verifiable
    T2_SOLID = 2       # Textbook equations, well-established
    T3_CONCEPTUAL = 3  # Estimated from analogy or scaling


@dataclass
class TieredValue:
    """A value with its data tier and source."""
    value: float
    tier: DataTier
    source: str = ""
    uncertainty_pct: float = 0.0  # estimated uncertainty (%)

    def __repr__(self):
        t = {DataTier.T1_KNOWN: "T1", DataTier.T2_SOLID: "T2",
             DataTier.T3_CONCEPTUAL: "T3"}[self.tier]
        return f"{self.value:.4g} [{t}]"


# ═══════════════════════════════════════════════════════════════════════════
# M5 Skeleton: Unified Material Design Interface
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TargetSpec:
    """Universal target specification for material design.

    Describes WHAT the material must capture/reflect/produce,
    independent of which material class is used.
    """
    name: str = ""
    target_type: str = ""         # "ion", "oxoanion", "molecule", "wavelength", "gas"

    # Capture targets
    target_species: str = ""      # e.g., "SeO3^2-", "Pb2+", "CO2"
    target_diameter_A: float = 0.0
    target_charge: int = 0
    target_mw: float = 0.0

    # Operating conditions
    pH: float = 7.0
    T_K: float = 298.15
    ionic_strength_M: float = 0.01
    solvent: str = "aqueous"

    # Performance targets
    target_capacity_mg_g: float = 0.0   # desired capacity (mg target / g material)
    target_selectivity: float = 0.0     # desired selectivity ratio vs interferent
    target_kinetics_min: float = 0.0    # desired equilibrium time (minutes)

    # Interferents
    interferent_species: List[str] = field(default_factory=list)
    interferent_concentrations_mM: List[float] = field(default_factory=list)

    # Optical targets (for structural color)
    target_wavelength_nm: float = 0.0
    target_angle_independent: bool = True

    # Scale
    required_scale: str = "lab"   # "lab", "pilot", "industrial"
    max_cost_per_kg: float = 0.0  # $/kg material budget


@dataclass
class PerformanceMetrics:
    """Predicted performance for a material design."""
    capacity_mg_g: TieredValue = None
    selectivity_ratio: TieredValue = None
    kinetics_t90_min: TieredValue = None  # time to 90% equilibrium
    cost_per_kg_usd: TieredValue = None
    scalability_score: TieredValue = None  # 0-1
    durability_cycles: TieredValue = None  # regeneration cycles
    environmental_score: TieredValue = None  # 0-1 (1=green)

    def composite_score(self, weights=None) -> float:
        """Weighted composite performance score.

        Higher = better. Normalizes each metric to 0-1 range.
        """
        if weights is None:
            weights = {"capacity": 0.3, "selectivity": 0.2,
                       "kinetics": 0.15, "cost": 0.15,
                       "scalability": 0.1, "environmental": 0.1}

        score = 0.0
        if self.capacity_mg_g:
            # Normalize: 100 mg/g = 1.0
            score += weights.get("capacity", 0) * min(1.0, self.capacity_mg_g.value / 100.0)
        if self.selectivity_ratio:
            # Normalize: 100x = 1.0
            score += weights.get("selectivity", 0) * min(1.0, self.selectivity_ratio.value / 100.0)
        if self.kinetics_t90_min:
            # Faster is better: 1 min = 1.0, 60 min = 0.1
            k = self.kinetics_t90_min.value
            score += weights.get("kinetics", 0) * (1.0 / (1.0 + k / 10.0))
        if self.cost_per_kg_usd:
            # Cheaper is better: $10/kg = 1.0, $1000/kg = 0.1
            c = self.cost_per_kg_usd.value
            score += weights.get("cost", 0) * (10.0 / (10.0 + c))
        if self.scalability_score:
            score += weights.get("scalability", 0) * self.scalability_score.value
        if self.environmental_score:
            score += weights.get("environmental", 0) * self.environmental_score.value

        return score


class MaterialDesign(ABC):
    """Abstract base for all material designs."""

    @abstractmethod
    def material_class(self) -> str:
        """Return material class label."""
        ...

    @abstractmethod
    def predict_performance(self, target: TargetSpec) -> PerformanceMetrics:
        """Predict performance for a given target."""
        ...

    @abstractmethod
    def design_summary(self) -> str:
        """Human-readable design description."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# M1: Porous Framework Design (MOFs, COFs, Zeolites)
# ═══════════════════════════════════════════════════════════════════════════

# ── Published data (T1) ───────────────────────────────────────────────────

# Common MOF/zeolite pore properties
# Sources: IZA database, CSD MOF subset, Moghadam et al. Chem. Mater. 2017
FRAMEWORK_DATABASE = {
    # name: (pore_diameter_A, BET_m2_g, pore_volume_cm3_g, density_g_cm3, cost_class)
    "HKUST-1":       (12.0, 1500, 0.75, 0.88, "moderate"),
    "UiO-66":        (11.0, 1200, 0.50, 1.24, "moderate"),
    "UiO-66-NH2":    (11.0, 1100, 0.47, 1.20, "moderate"),
    "ZIF-8":         (11.6, 1630, 0.66, 0.95, "low"),
    "MOF-5":         (15.1, 3800, 1.55, 0.59, "moderate"),
    "MIL-101(Cr)":   (34.0, 4100, 2.00, 0.44, "high"),
    "MOF-74(Zn)":    (10.8,  816, 0.39, 1.23, "moderate"),
    "MIL-53(Al)":    ( 8.5,  900, 0.50, 0.98, "low"),
    "NU-1000":       (12.0, 2200, 1.40, 0.49, "high"),
    "COF-300":       (10.0, 1360, 0.80, 0.65, "high"),
    "COF-5":         (27.0,  780, 1.00, 0.58, "high"),
    "Zeolite-Y":     (13.0,  900, 0.35, 1.27, "low"),
    "ZSM-5":         ( 5.5,  400, 0.18, 1.80, "low"),
    "SAPO-34":       ( 3.8,  600, 0.27, 1.65, "low"),
    "BEA":           ( 7.5,  620, 0.25, 1.56, "low"),
}

# Common functional groups for MOF/COF post-synthetic modification
# (group_name, donor_types, pKa_relevant, selectivity_target)
FUNCTIONAL_GROUPS = {
    "amine-primary":   (["N_amine"], 10.0, "CO2, heavy metals"),
    "carboxylate":     (["O_carboxylate"], 4.0, "heavy metals, cations"),
    "thiol":           (["S_thiol"], 8.5, "soft metals (Hg, Pb, Cd)"),
    "phosphonate":     (["O_phosphate"], 6.5, "oxoanions, actinides"),
    "hydroxyl":        (["O_hydroxyl"], 15.7, "general H-bond"),
    "pyridyl":         (["N_pyridine"], 5.2, "transition metals"),
    "amidoxime":       (["N_imine", "O_hydroxyl", "N_amine"], 6.0, "uranium, oxoanions"),
    "sulfonate":       (["O_sulfonate"], -2.0, "cations, ion exchange"),
    "quaternary-N":    ([], None, "anions, ion exchange"),
}

# Cost estimates (T3)
COST_CLASS = {
    "low": TieredValue(50.0, DataTier.T3_CONCEPTUAL, "zeolite/ZIF bulk: ~$50/kg", 50),
    "moderate": TieredValue(200.0, DataTier.T3_CONCEPTUAL, "UiO/HKUST lab scale: ~$200/kg", 50),
    "high": TieredValue(1000.0, DataTier.T3_CONCEPTUAL, "NU-1000/COF research: ~$1000/kg", 100),
}


@dataclass
class PorousFrameworkSpec:
    """Specification for a porous framework material design."""
    framework_name: str = ""          # known framework or "custom"
    framework_type: str = "MOF"       # MOF, COF, zeolite

    # Structural properties
    pore_diameter_A: float = 0.0
    surface_area_m2_g: float = 0.0
    pore_volume_cm3_g: float = 0.0
    density_g_cm3: float = 1.0

    # Functionalization
    functional_groups: List[str] = field(default_factory=list)
    func_group_density_mmol_g: float = 0.0  # loading density

    # Stability
    water_stable: bool = True
    pH_range: Tuple[float, float] = (2.0, 12.0)
    max_temp_K: float = 523.15        # 250°C default
    regenerable: bool = True

    # Cost
    cost_class: str = "moderate"

    @classmethod
    def from_database(cls, name: str) -> 'PorousFrameworkSpec':
        """Create spec from known framework database."""
        if name not in FRAMEWORK_DATABASE:
            raise KeyError(f"Unknown framework '{name}'. "
                           f"Known: {sorted(FRAMEWORK_DATABASE.keys())}")
        pore_d, bet, pv, rho, cost = FRAMEWORK_DATABASE[name]
        ftype = "zeolite" if name in ("Zeolite-Y", "ZSM-5", "SAPO-34", "BEA") else \
                "COF" if "COF" in name else "MOF"
        return cls(
            framework_name=name, framework_type=ftype,
            pore_diameter_A=pore_d, surface_area_m2_g=bet,
            pore_volume_cm3_g=pv, density_g_cm3=rho,
            cost_class=cost,
        )


# ── Adsorption Physics (T2) ──────────────────────────────────────────────

def langmuir_capacity(q_max_mg_g: float, K_L: float, C_eq_mM: float) -> float:
    """Langmuir isotherm: equilibrium adsorption capacity.

    q = q_max × K_L × C / (1 + K_L × C)

    Parameters
    ----------
    q_max_mg_g : float
        Maximum monolayer capacity (mg/g).
    K_L : float
        Langmuir constant (L/mmol).
    C_eq_mM : float
        Equilibrium concentration (mM).

    Returns
    -------
    float
        Equilibrium capacity (mg/g).

    Physics tier: T2 (Langmuir 1918, standard textbook).
    """
    C = C_eq_mM  # mM = mmol/L
    return q_max_mg_g * K_L * C / (1.0 + K_L * C)


def langmuir_q_max_from_sites(site_density_mmol_g: float,
                                target_mw: float) -> float:
    """Estimate maximum capacity from binding site density.

    q_max = site_density × MW_target (converting mmol sites/g to mg target/g)

    Parameters
    ----------
    site_density_mmol_g : float
        Binding sites per gram of material (mmol/g).
    target_mw : float
        Molecular weight of target species (g/mol).

    Returns
    -------
    float
        Maximum capacity in mg/g.

    Physics tier: T2 (stoichiometry).
    """
    return site_density_mmol_g * target_mw


def langmuir_K_from_dG(dG_bind_kJ: float, T: float = T_STD) -> float:
    """Langmuir constant from binding free energy.

    K_L = exp(-ΔG_bind / RT)

    Note: K_L is dimensionless equilibrium constant. For Langmuir
    in concentration units, K_L ≈ K_eq / C_ref where C_ref = 1 M.

    Parameters
    ----------
    dG_bind_kJ : float
        Binding free energy (kJ/mol). Negative = favorable.
    T : float
        Temperature (K).

    Returns
    -------
    float
        Langmuir constant (L/mmol, assuming C_ref normalization).

    Physics tier: T2 (thermodynamics).
    """
    RT = R_kJ * T
    K_eq = math.exp(-dG_bind_kJ / RT)
    # Convert from dimensionless to L/mmol (1 M reference → 1000 mmol/L)
    return K_eq / 1000.0


def selectivity_from_K(K_target: float, K_interferent: float) -> float:
    """Selectivity ratio from Langmuir constants.

    α = K_target / K_interferent

    Parameters
    ----------
    K_target : float
        Langmuir constant for target species.
    K_interferent : float
        Langmuir constant for interferent species.

    Returns
    -------
    float
        Selectivity ratio (>1 = selective for target).

    Physics tier: T2 (competitive Langmuir).
    """
    if K_interferent <= 0:
        return float('inf')
    return K_target / K_interferent



# ── Competitive Langmuir (T2) ────────────────────────────────────────────

def competitive_langmuir(q_max_mg_g: float, K_target: float, C_target_mM: float,
                          K_interferents: list, C_interferents_mM: list) -> float:
    """Competitive (multi-component) Langmuir isotherm.

    q_target = q_max × K_t × C_t / (1 + K_t × C_t + Σ_i K_i × C_i)

    All species compete for the same sites. Higher K_i × C_i for
    interferents reduces target uptake.

    Parameters
    ----------
    q_max_mg_g : float
        Maximum monolayer capacity (mg target / g sorbent).
    K_target : float
        Langmuir constant for target (L/mmol).
    C_target_mM : float
        Target equilibrium concentration (mM).
    K_interferents : list of float
        Langmuir constants for each interferent (L/mmol).
    C_interferents_mM : list of float
        Equilibrium concentrations of interferents (mM).

    Returns
    -------
    float
        Target uptake under competition (mg/g).

    Physics tier: T2 (Butler & Ockrent 1930; Markham & Benton 1931).
    """
    denominator = 1.0 + K_target * C_target_mM
    for Ki, Ci in zip(K_interferents, C_interferents_mM):
        denominator += Ki * Ci
    if denominator <= 0:
        return 0.0
    return q_max_mg_g * K_target * C_target_mM / denominator


def competitive_reduction_factor(K_target: float, C_target_mM: float,
                                   K_interferents: list,
                                   C_interferents_mM: list) -> float:
    """Fraction of single-component capacity retained under competition.

    f = q_competitive / q_single = (1 + K_t×C_t) / (1 + K_t×C_t + Σ K_i×C_i)

    Parameters
    ----------
    (same as competitive_langmuir)

    Returns
    -------
    float
        Reduction factor 0–1. 1.0 = no competition effect.

    Physics tier: T2.
    """
    single_denom = 1.0 + K_target * C_target_mM
    comp_denom = single_denom
    for Ki, Ci in zip(K_interferents, C_interferents_mM):
        comp_denom += Ki * Ci
    if comp_denom <= 0:
        return 0.0
    return single_denom / comp_denom


# ── Ionic Strength Correction: Davies Equation (T2) ──────────────────────

def davies_log_gamma(z: int, I: float) -> float:
    """Davies equation: activity coefficient for an ion.

    log₁₀(γ) = -A × z² × (√I / (1 + √I) - 0.3 × I)

    where A = 0.509 at 25°C in water.

    Valid for I < 0.5 M. Standard in environmental/geochemistry.

    Parameters
    ----------
    z : int
        Ion charge (absolute value used).
    I : float
        Ionic strength (M).

    Returns
    -------
    float
        log₁₀(γ), typically negative (γ < 1 at I > 0).

    Physics tier: T2 (Davies 1962, based on Debye-Hückel theory).

    Reference:
        Davies CW. Ion Association. Butterworths, London, 1962.
    """
    A = 0.509  # Debye-Hückel A parameter at 25°C, water
    z_abs = abs(z)
    if I <= 0:
        return 0.0  # ideal solution
    sqrt_I = math.sqrt(I)
    return -A * z_abs**2 * (sqrt_I / (1.0 + sqrt_I) - 0.3 * I)


def activity_coefficient(z: int, I: float) -> float:
    """Activity coefficient γ from Davies equation.

    γ = 10^(log₁₀(γ))

    Physics tier: T2.
    """
    return 10.0 ** davies_log_gamma(z, I)


def ionic_strength_from_species(charges: list, concentrations_mM: list) -> float:
    """Compute ionic strength from solution composition.

    I = 0.5 × Σ c_i × z_i²

    Parameters
    ----------
    charges : list of int
    concentrations_mM : list of float
        In mM (= mmol/L = mol/m³).

    Returns
    -------
    float
        Ionic strength in M (mol/L).

    Physics tier: T2 (Lewis & Randall 1921).
    """
    I = 0.0
    for z, c in zip(charges, concentrations_mM):
        I += c * z**2
    return I * 0.5 / 1000.0  # mM → M


def correct_K_for_ionic_strength(K_ideal: float, z_target: int,
                                   z_site: int, I: float) -> float:
    """Correct a Langmuir constant for non-ideal solution behavior.

    The binding equilibrium A^z + S^z' ⇌ AS involves activity coefficients:
    K_apparent = K_ideal × γ_A × γ_S / γ_AS

    Simplified for sorbent sites (γ_S ≈ 1, γ_AS ≈ 1):
    K_apparent ≈ K_ideal × γ_A

    Higher ionic strength → lower γ → lower effective K for charged species.
    Neutral species (z=0): γ = 1, no correction.

    Parameters
    ----------
    K_ideal : float
        Langmuir constant at I → 0.
    z_target : int
        Target ion charge.
    z_site : int
        Sorbent site charge (for product correction). Set to 0 if unknown.
    I : float
        Ionic strength (M).

    Returns
    -------
    float
        Corrected Langmuir constant.

    Physics tier: T2.
    """
    if I <= 0 or z_target == 0:
        return K_ideal
    gamma_target = activity_coefficient(z_target, I)
    # Product correction: AS complex has charge z_target + z_site
    gamma_product = activity_coefficient(z_target + z_site, I) if z_site != 0 else 1.0
    gamma_site = activity_coefficient(z_site, I) if z_site != 0 else 1.0
    # K_app = K_ideal × γ_A × γ_S / γ_AS
    if gamma_product <= 0:
        return K_ideal
    return K_ideal * gamma_target * gamma_site / gamma_product


# ── Thomas Model Breakthrough Curve (T2) ─────────────────────────────────

def thomas_breakthrough(BV: float, BV_50: float, k_Th_BV: float) -> float:
    """Thomas model: effluent concentration ratio at a given bed volume.

    C/C₀ = 1 / (1 + exp(k_Th × (BV_50 - BV)))

    This is the sigmoidal breakthrough curve. At BV = BV_50, C/C₀ = 0.5.
    k_Th controls the steepness (sharper front = higher k_Th).

    Parameters
    ----------
    BV : float
        Bed volumes of solution passed (dimensionless).
    BV_50 : float
        Bed volumes at 50% breakthrough (center of S-curve).
    k_Th_BV : float
        Thomas rate parameter (1/BV). Higher = sharper front.
        Typical: 0.001–0.01 for IX, 0.01–0.1 for adsorption.

    Returns
    -------
    float
        C/C₀ at the given BV (0–1).

    Physics tier: T2 (Thomas 1944).

    Reference:
        Thomas HC. J. Am. Chem. Soc. 1944, 66, 1664.
    """
    exponent = k_Th_BV * (BV_50 - BV)
    # Clamp to avoid overflow
    exponent = max(-500.0, min(500.0, exponent))
    return 1.0 / (1.0 + math.exp(exponent))


def thomas_BV_at_breakthrough(BV_50: float, k_Th_BV: float,
                                C_ratio: float = 0.05) -> float:
    """Bed volumes at a specified breakthrough ratio.

    Inverts the Thomas model:
    BV = BV_50 - ln(1/f - 1) / k_Th

    where f = C/C₀ at breakthrough.

    Parameters
    ----------
    BV_50 : float
        Bed volumes at 50% breakthrough.
    k_Th_BV : float
        Thomas rate parameter (1/BV).
    C_ratio : float
        Breakthrough criterion (default 0.05 = 5% of feed).

    Returns
    -------
    float
        Bed volumes to specified breakthrough.

    Physics tier: T2.
    """
    if k_Th_BV <= 0 or C_ratio <= 0 or C_ratio >= 1:
        return BV_50
    return BV_50 - math.log(1.0 / C_ratio - 1.0) / k_Th_BV


def thomas_curve(BV_50: float, k_Th_BV: float,
                  n_points: int = 100) -> list:
    """Generate a full Thomas breakthrough curve.

    Returns list of (BV, C/C₀) tuples from 0 to 2×BV_50.

    Physics tier: T2.
    """
    BV_max = 2.0 * BV_50
    curve = []
    for i in range(n_points + 1):
        BV = BV_max * i / n_points
        ratio = thomas_breakthrough(BV, BV_50, k_Th_BV)
        curve.append((BV, ratio))
    return curve


def estimate_thomas_k_from_kinetics(t90_min: float, v_m_s: float,
                                      dp_m: float) -> float:
    """Estimate Thomas rate parameter from kinetic and flow data.

    k_Th ≈ 1 / (t_90 × v / dp) (dimensionless scaling)

    This is a T2 dimensional analysis estimate, not fitted.

    Parameters
    ----------
    t90_min : float
        Time to 90% equilibrium in batch (minutes).
    v_m_s : float
        Superficial velocity (m/s).
    dp_m : float
        Particle diameter (m).

    Returns
    -------
    float
        Estimated k_Th in 1/BV units.

    Physics tier: T2 (dimensional analysis).
    """
    if t90_min <= 0 or v_m_s <= 0 or dp_m <= 0:
        return 0.01  # default moderate steepness
    t90_s = t90_min * 60.0
    # Dimensionless time scale: number of particle transits during t90
    n_transits = v_m_s * t90_s / dp_m
    if n_transits <= 0:
        return 0.01
    # k_Th scales inversely with the number of transits needed
    return 1.0 / n_transits



def pore_size_exclusion(pore_diameter_A: float,
                         species_diameter_A: float) -> bool:
    """Check if species can enter the pore.

    Species needs pore_diameter > species_diameter + 2×vdW_clearance.
    vdW clearance ≈ 0.5 Å minimum for passage.

    Physics tier: T2 (steric exclusion).
    """
    VDW_CLEARANCE = 0.5  # Å minimum per side
    return pore_diameter_A >= species_diameter_A + 2 * VDW_CLEARANCE


def estimate_site_density(surface_area_m2_g: float,
                           func_group_density_per_nm2: float = 1.0) -> float:
    """Estimate binding site density from surface area.

    site_density (mmol/g) = SA (m²/g) × group_density (nm⁻²) × 10⁶ (nm²/m²) / N_A

    Parameters
    ----------
    surface_area_m2_g : float
        BET surface area (m²/g).
    func_group_density_per_nm2 : float
        Functional groups per nm² of surface. Typical: 0.5-3.0.
        Default 1.0 is conservative for post-synthetic modification.

    Returns
    -------
    float
        Site density in mmol/g.

    Physics tier: T3 (estimated from typical MOF functionalization densities).
    """
    # SA in m²/g → nm²/g: ×10^18
    sites_per_g = surface_area_m2_g * 1e18 * func_group_density_per_nm2
    return sites_per_g / N_AVO * 1e3  # convert to mmol/g


def estimate_kinetics_t90(pore_diameter_A: float,
                           particle_size_um: float = 10.0,
                           D_eff_scale: float = 0.1) -> float:
    """Estimate time to 90% equilibrium (minutes).

    Based on intraparticle diffusion model:
    t_90 ≈ R² / (π² × D_eff)

    where R = particle radius, D_eff = effective diffusivity.
    D_eff ≈ D_bulk × (pore_d/species_d)² × porosity / tortuosity

    Simplified: D_eff ≈ D_bulk × D_eff_scale
    D_bulk for small ions in water ≈ 1e-9 m²/s

    Parameters
    ----------
    pore_diameter_A : float
        Pore diameter (Å). Larger pores = faster diffusion.
    particle_size_um : float
        Particle diameter (μm).
    D_eff_scale : float
        Fraction of bulk diffusivity (accounts for tortuosity, constriction).

    Returns
    -------
    float
        Estimated t_90 in minutes.

    Physics tier: T3 (simplified Fick's law with estimated tortuosity).
    """
    D_bulk = 1e-9  # m²/s for small ions
    # Constriction factor: small pores restrict diffusion
    if pore_diameter_A > 0:
        constriction = min(1.0, (pore_diameter_A / 5.0) ** 2)
    else:
        constriction = 0.1
    D_eff = D_bulk * D_eff_scale * constriction

    R_m = particle_size_um * 1e-6 / 2.0  # radius in meters
    if D_eff <= 0:
        return 999.0
    t_90_s = R_m ** 2 / (PI ** 2 * D_eff)
    return t_90_s / 60.0  # seconds → minutes


# ── pKa integration ──────────────────────────────────────────────────────

def func_group_fraction_active(group_name: str, pH: float,
                                for_anion_capture: bool = False) -> float:
    """Fraction of a functional group active at given pH.

    For METAL coordination: active form is deprotonated (lone pair available).
        Carboxylate as COO⁻, thiol as RS⁻, amine as RNH₂.
    For ANION capture: basic groups are active when PROTONATED (cationic).
        Amine as RNH₃⁺ (binds anions), carboxylate still needs COO⁻.

    Parameters
    ----------
    group_name : str
        Key from FUNCTIONAL_GROUPS.
    pH : float
        Working pH.
    for_anion_capture : bool
        If True, basic groups (amine, pyridyl) are active when protonated.

    Returns
    -------
    float
        Fraction active (0-1).
    """
    if group_name not in FUNCTIONAL_GROUPS:
        return 1.0

    donors, pKa, _ = FUNCTIONAL_GROUPS[group_name]
    if pKa is None or pKa < -1.0:
        return 1.0  # always active (e.g., sulfonate, quaternary N)

    # Basic groups (amine, pyridyl): active form depends on application
    BASIC_GROUPS = {"amine-primary", "pyridyl"}

    if for_anion_capture and group_name in BASIC_GROUPS:
        # For anion capture: protonated (cationic) form is active
        # f_protonated = 1 / (1 + 10^(pH - pKa))
        return 1.0 / (1.0 + 10.0 ** (pH - pKa))

    # Default: deprotonated form is active (coordination, H-bond acceptance)
    return 1.0 / (1.0 + 10.0 ** (pKa - pH))



# ═══════════════════════════════════════════════════════════════════════════
# Repulsion Integration Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _framework_to_site_spec(spec) -> 'MaterialSiteSpec':
    """Map a PorousFrameworkSpec to a MaterialSiteSpec for repulsion scoring.

    Derives cavity radius, donor types, hydrophobicity, CN from
    framework properties. No estimation — maps directly.
    """
    from core.repulsion_physics import MaterialSiteSpec

    # Cavity radius from pore diameter (pore = accessible channel, not cavity)
    # Binding site is at the pore wall; effective radius ≈ pore_d / 3
    cavity_r = spec.pore_diameter_A / 3.0 if spec.pore_diameter_A > 0 else 3.0

    # Donor types from functional groups
    donor_types = []
    for fg in spec.functional_groups:
        fg_info = FUNCTIONAL_GROUPS.get(fg)
        if fg_info:
            donor_types.extend(fg_info[0])

    # Hydrophobicity: MOFs with organic linkers are moderately hydrophobic
    hydro_map = {"MOF": 0.4, "COF": 0.5, "zeolite": 0.2}
    hydrophobicity = hydro_map.get(spec.framework_type, 0.3)

    # CN: most framework sites are 4-6 coordinate
    cn = 6 if spec.framework_type == "zeolite" else 4

    return MaterialSiteSpec(
        cavity_radius_A=cavity_r,
        site_charge=0,
        hydrophobicity=hydrophobicity,
        donor_types=donor_types,
        offered_CN=cn,
        offered_geometry="octahedral" if cn >= 6 else "tetrahedral",
    )


def _polymer_to_site_spec(spec) -> 'MaterialSiteSpec':
    """Map a PolymericSorbentSpec to a MaterialSiteSpec for repulsion scoring."""
    from core.repulsion_physics import MaterialSiteSpec

    # Resin bead interior — no well-defined cavity
    # Use approximate pore size from resin type
    cavity_r_map = {
        "SAC": 4.0, "WAC": 4.0, "SBA": 4.0, "WBA": 4.0,
        "chelating": 3.0, "specialty": 3.5, "hydrogel": 5.0,
    }
    cavity_r = cavity_r_map.get(spec.resin_type, 4.0)

    # Site charge from resin type
    charge_map = {
        "SAC": -1, "WAC": -1, "SBA": 1, "WBA": 1,
        "chelating": 0, "specialty": 0, "hydrogel": 0,
    }
    site_charge = charge_map.get(spec.resin_type, 0)

    # Donor types from functional group
    donor_types = []
    fg_info = FUNCTIONAL_GROUPS.get(spec.functional_group)
    if fg_info:
        donor_types = list(fg_info[0])
    # Chelating: iminodiacetate has N + 2×O
    if spec.functional_group == "iminodiacetate":
        donor_types = ["N_amine", "O_carboxylate", "O_carboxylate"]
    elif spec.functional_group == "thiol":
        donor_types = ["S_thiol"]
    elif spec.functional_group == "amidoxime":
        donor_types = ["N_imine", "O_hydroxyl", "N_amine"]

    # Hydrophobicity from matrix
    hydro_map = {
        "PS-DVB": 0.5, "polyacrylic": 0.2, "chitosan": 0.15,
        "polyethylene": 0.6, "crosslinked-HEMA": 0.2,
    }
    hydrophobicity = hydro_map.get(spec.matrix, 0.3)

    # Fixed charge for Donnan
    fixed_charge = spec.capacity_meq_g * 0.5 if spec.resin_type in ("SAC", "SBA") else 0.0

    return MaterialSiteSpec(
        cavity_radius_A=cavity_r,
        site_charge=site_charge,
        fixed_charge_meq_mL=fixed_charge,
        hydrophobicity=hydrophobicity,
        donor_types=donor_types,
        offered_CN=6,
        offered_geometry="octahedral",
    )


def _compute_repulsion_selectivity(site_spec, target_species: str,
                                     interferent_species: list,
                                     dG_attract: float = -25.0) -> float:
    """Compute worst-case selectivity from differential repulsion.

    Returns the minimum selectivity ratio across all interferents.
    """
    from core.repulsion_physics import selectivity_from_differential_repulsion

    if not interferent_species:
        return 1.0

    worst_log_alpha = float('inf')
    for intf in interferent_species:
        sd = selectivity_from_differential_repulsion(
            site_spec, target_species, intf,
            dG_attract, dG_attract,
        )
        worst_log_alpha = min(worst_log_alpha, sd.log_selectivity)

    # Convert log selectivity to ratio
    if worst_log_alpha == float('inf'):
        return 1.0
    return 10.0 ** worst_log_alpha


# ── Framework Design Engine ──────────────────────────────────────────────

@dataclass
class FrameworkDesign(MaterialDesign):
    """Complete porous framework material design."""
    spec: PorousFrameworkSpec = field(default_factory=PorousFrameworkSpec)
    target: TargetSpec = field(default_factory=TargetSpec)
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)

    # Design details
    q_max_mg_g: float = 0.0
    K_L: float = 0.0
    site_density_mmol_g: float = 0.0
    pore_accessible: bool = True
    func_group_activity: Dict[str, float] = field(default_factory=dict)

    def material_class(self) -> str:
        return f"PorousFramework ({self.spec.framework_type})"

    def design_summary(self) -> str:
        lines = [
            f"Framework: {self.spec.framework_name} ({self.spec.framework_type})",
            f"Pore: {self.spec.pore_diameter_A:.1f} Å, "
            f"SA: {self.spec.surface_area_m2_g:.0f} m²/g",
        ]
        if self.spec.functional_groups:
            lines.append(f"Functionalization: {', '.join(self.spec.functional_groups)} "
                         f"@ {self.spec.func_group_density_mmol_g:.2f} mmol/g")
        if self.metrics.capacity_mg_g:
            lines.append(f"Predicted capacity: {self.metrics.capacity_mg_g}")
        if self.metrics.selectivity_ratio:
            lines.append(f"Predicted selectivity: {self.metrics.selectivity_ratio}")
        return "\n".join(lines)

    def predict_performance(self, target: TargetSpec = None) -> PerformanceMetrics:
        if target is None:
            target = self.target

        self.target = target
        s = self.spec

        # 1. Pore accessibility check
        self.pore_accessible = pore_size_exclusion(
            s.pore_diameter_A, target.target_diameter_A
        )

        # 2. Site density (T3 estimate from surface area + loading)
        if s.func_group_density_mmol_g > 0:
            self.site_density_mmol_g = s.func_group_density_mmol_g
        else:
            # Estimate from surface area
            self.site_density_mmol_g = estimate_site_density(
                s.surface_area_m2_g, func_group_density_per_nm2=1.0
            )

        # 3. Functional group activity at working pH
        effective_site_density = self.site_density_mmol_g
        is_anion_target = target.target_charge < 0
        for fg in s.functional_groups:
            f_active = func_group_fraction_active(
                fg, target.pH, for_anion_capture=is_anion_target
            )
            self.func_group_activity[fg] = f_active
            effective_site_density *= f_active

        # 4. Maximum capacity (T2: stoichiometric)
        mw = target.target_mw if target.target_mw > 0 else 80.0  # default ~ SeO3
        self.q_max_mg_g = langmuir_q_max_from_sites(effective_site_density, mw)

        # If pore is inaccessible, capacity drops to external surface only (~5%)
        if not self.pore_accessible:
            self.q_max_mg_g *= 0.05

        # 5. Langmuir K from estimated binding energy (T3)
        # Rough: -20 to -40 kJ/mol for typical MOF-guest interactions
        dG_bind = -25.0  # kJ/mol default (moderate affinity)
        if target.target_charge != 0 and s.functional_groups:
            dG_bind = -35.0  # charged target + functionalized = stronger
        self.K_L = langmuir_K_from_dG(dG_bind, target.T_K)

        # 6. Ionic strength correction (T2: Davies equation)
        I = target.ionic_strength_M
        if I > 0 and target.target_charge != 0:
            self.K_L = correct_K_for_ionic_strength(
                self.K_L, target.target_charge, 0, I
            )

        # 7. Equilibrium capacity — single component
        C_eq = 0.1  # mM default (typical trace contaminant)
        q_eq_single = langmuir_capacity(self.q_max_mg_g, self.K_L, C_eq)

        # 8. Competitive Langmuir if interferents present
        q_eq = q_eq_single
        comp_factor = 1.0
        if target.interferent_species and target.interferent_concentrations_mM:
            # Estimate K for interferents (scaled from target K by charge ratio)
            K_intfs = []
            C_intfs = list(target.interferent_concentrations_mM)
            for intf in target.interferent_species:
                # Interferent K: assume similar binding but corrected for charge
                K_i = self.K_L * 0.5  # default: interferent binds half as well
                if I > 0:
                    # Rough charge estimate from species string
                    z_i = intf.count('+') - intf.count('-')
                    if z_i == 0:
                        z_i = 1  # default monovalent
                    K_i = correct_K_for_ionic_strength(K_i, z_i, 0, I)
                K_intfs.append(K_i)

            # Pad concentrations if fewer than species
            while len(C_intfs) < len(K_intfs):
                C_intfs.append(1.0)  # 1 mM default interferent

            q_eq = competitive_langmuir(
                self.q_max_mg_g, self.K_L, C_eq, K_intfs, C_intfs
            )
            comp_factor = competitive_reduction_factor(
                self.K_L, C_eq, K_intfs, C_intfs
            )

        # 9. Selectivity from differential repulsion (physics-based)
        selectivity = 1.0
        sel_tier = DataTier.T2_SOLID
        sel_source = "repulsion physics"
        if not self.pore_accessible:
            selectivity = 0.1
            sel_source = "pore inaccessible"
        elif target.interferent_species:
            try:
                site_spec = _framework_to_site_spec(s)
                selectivity = _compute_repulsion_selectivity(
                    site_spec, target.target_species,
                    target.interferent_species,
                )
                # Clamp to reasonable range
                selectivity = max(0.01, min(1e6, selectivity))
                sel_source = "differential repulsion (R1-R5)"
            except (ImportError, Exception):
                # Fallback if repulsion module unavailable
                selectivity = 10.0 if target.target_charge != 0 else 1.0
                sel_tier = DataTier.T3_CONCEPTUAL
                sel_source = "heuristic fallback"

        # 8. Kinetics
        # Realistic bead size for column application (200 μm default)
        t90 = estimate_kinetics_t90(s.pore_diameter_A, particle_size_um=200.0)

        # 9. Cost
        cost = COST_CLASS.get(s.cost_class, COST_CLASS["moderate"])

        # 10. Scalability
        scale_map = {
            "zeolite": 0.9,  # industrial process
            "MOF": 0.5,      # pilot scale demonstrated for some
            "COF": 0.2,      # mostly lab scale
        }
        scalability = scale_map.get(s.framework_type, 0.3)

        # 11. Environmental score
        env_map = {
            "zeolite": 0.8,  # natural mineral or simple synthesis
            "MOF": 0.5,      # solvent-intensive synthesis
            "COF": 0.4,      # requires specific monomers
        }
        env = env_map.get(s.framework_type, 0.5)

        self.metrics = PerformanceMetrics(
            capacity_mg_g=TieredValue(q_eq, DataTier.T2_SOLID,
                f"Langmuir: q_max={self.q_max_mg_g:.1f} mg/g, K_L={self.K_L:.4f}", 30),
            selectivity_ratio=TieredValue(selectivity, sel_tier,
                sel_source, 30),
            kinetics_t90_min=TieredValue(t90, DataTier.T3_CONCEPTUAL,
                "Intraparticle diffusion estimate", 50),
            cost_per_kg_usd=cost,
            scalability_score=TieredValue(scalability, DataTier.T3_CONCEPTUAL,
                f"{s.framework_type} scalability estimate"),
            durability_cycles=TieredValue(
                50.0 if s.regenerable else 1.0, DataTier.T3_CONCEPTUAL,
                "Regeneration cycle estimate"),
            environmental_score=TieredValue(env, DataTier.T3_CONCEPTUAL,
                f"{s.framework_type} environmental impact"),
        )

        return self.metrics


def design_framework(framework_name: str, target: TargetSpec,
                      functional_groups: Optional[List[str]] = None,
                      func_density_mmol_g: float = 0.0) -> FrameworkDesign:
    """Design a porous framework material for a target.

    Parameters
    ----------
    framework_name : str
        Known framework name (from FRAMEWORK_DATABASE) or "custom".
    target : TargetSpec
        What to capture.
    functional_groups : list of str, optional
        Functional groups for post-synthetic modification.
    func_density_mmol_g : float
        Functionalization density (mmol/g). 0 = auto-estimate.

    Returns
    -------
    FrameworkDesign
        Complete design with predicted performance.
    """
    spec = PorousFrameworkSpec.from_database(framework_name)
    if functional_groups:
        spec.functional_groups = functional_groups
    if func_density_mmol_g > 0:
        spec.func_group_density_mmol_g = func_density_mmol_g

    design = FrameworkDesign(spec=spec)
    design.predict_performance(target)
    return design


def screen_frameworks(target: TargetSpec,
                       functional_groups: Optional[List[str]] = None,
                       top_n: int = 10) -> List[FrameworkDesign]:
    """Screen all known frameworks against a target.

    Returns ranked list (best composite score first).
    """
    designs = []
    for name in FRAMEWORK_DATABASE:
        try:
            d = design_framework(name, target,
                                  functional_groups=functional_groups)
            designs.append(d)
        except Exception:
            pass

    designs.sort(key=lambda d: d.metrics.composite_score(), reverse=True)
    return designs[:top_n]


# ═══════════════════════════════════════════════════════════════════════════
# M5: Unified Designer — Cross-Material Ranking
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MaterialRanking:
    """One material design in the cross-material ranking."""
    material_class: str
    design_name: str
    design: MaterialDesign
    metrics: PerformanceMetrics
    composite_score: float = 0.0
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)


@dataclass
class UnifiedDesignResult:
    """Complete cross-material design comparison."""
    target: TargetSpec
    rankings: List[MaterialRanking] = field(default_factory=list)
    n_materials_evaluated: int = 0
    best_for_capacity: str = ""
    best_for_cost: str = ""
    best_for_selectivity: str = ""
    best_for_scalability: str = ""


def unified_design(target: TargetSpec,
                    include_frameworks: bool = True,
                    include_polymers: bool = True,      # M2: Polymeric Sorbents
                    include_composites: bool = True,     # M3: Composite Materials
                    include_structural_color: bool = True,   # M4: Structural Color
                    framework_func_groups: Optional[List[str]] = None,
                    top_n: int = 15) -> UnifiedDesignResult:
    """Cross-material design comparison.

    Evaluates all material classes against the same target,
    ranks by composite performance score.

    Currently implements: M1 (PorousFramework).
    M2-M4 stubs ready for future sessions.
    """
    all_rankings = []

    # M1: Porous Frameworks
    if include_frameworks:
        frameworks = screen_frameworks(target, framework_func_groups)
        for d in frameworks:
            strengths, weaknesses = [], []
            m = d.metrics
            if m.capacity_mg_g and m.capacity_mg_g.value > 50:
                strengths.append("high capacity")
            if m.cost_per_kg_usd and m.cost_per_kg_usd.value < 100:
                strengths.append("low cost")
            if m.scalability_score and m.scalability_score.value > 0.7:
                strengths.append("industrially scalable")
            if not d.pore_accessible:
                weaknesses.append("pore too small for target")
            if m.kinetics_t90_min and m.kinetics_t90_min.value > 30:
                weaknesses.append("slow kinetics")
            if m.cost_per_kg_usd and m.cost_per_kg_usd.value > 500:
                weaknesses.append("expensive")

            all_rankings.append(MaterialRanking(
                material_class=d.material_class(),
                design_name=d.spec.framework_name,
                design=d,
                metrics=m,
                composite_score=m.composite_score(),
                strengths=strengths,
                weaknesses=weaknesses,
            ))

    # M2: Polymeric Sorbents
    if include_polymers:
        polymers = screen_polymers(target)
        for d in polymers:
            strengths, weaknesses = [], []
            m = d.metrics
            if m.capacity_mg_g and m.capacity_mg_g.value > 50:
                strengths.append("high capacity")
            if m.cost_per_kg_usd and m.cost_per_kg_usd.value < 30:
                strengths.append("very low cost")
            if m.scalability_score and m.scalability_score.value > 0.8:
                strengths.append("industrially mature")
            if m.durability_cycles and m.durability_cycles.value > 100:
                strengths.append("regenerable")
            if m.selectivity_ratio and m.selectivity_ratio.value < 1.0:
                weaknesses.append("poor selectivity for target")
            if m.kinetics_t90_min and m.kinetics_t90_min.value > 30:
                weaknesses.append("slow kinetics")
            if d.func_group_activity < 0.1:
                weaknesses.append(f"func group inactive at pH {target.pH}")

            all_rankings.append(MaterialRanking(
                material_class=d.material_class(),
                design_name=d.spec.resin_name,
                design=d,
                metrics=m,
                composite_score=m.composite_score(),
                strengths=strengths,
                weaknesses=weaknesses,
            ))

    # M3: Composite Materials
    if include_composites:
        composites = screen_composites(target)
        for d in composites:
            strengths, weaknesses = [], []
            m = d.metrics
            if m.capacity_mg_g and m.capacity_mg_g.value > 5:
                strengths.append("published capacity")
            if d.BV_to_breakthrough and d.BV_to_breakthrough > 5000:
                strengths.append(f"long run ({d.BV_to_breakthrough:.0f} BV)")
            if m.scalability_score and m.scalability_score.value > 0.7:
                strengths.append("scalable process")
            if d.pressure_drop_kPa and d.pressure_drop_kPa < 10:
                strengths.append("low pressure drop")
            if m.capacity_mg_g is None:
                weaknesses.append("no capacity data for this target")
            if d.pressure_drop_kPa and d.pressure_drop_kPa > 50:
                weaknesses.append("high pressure drop")

            all_rankings.append(MaterialRanking(
                material_class=d.material_class(),
                design_name=d.config.name,
                design=d,
                metrics=m,
                composite_score=m.composite_score(),
                strengths=strengths,
                weaknesses=weaknesses,
            ))

    # M4: Structural Color
    if include_structural_color and target.target_wavelength_nm > 0:
        sc_designs = screen_structural_colors(target)
        for d in sc_designs:
            strengths, weaknesses = [], []
            m = d.metrics
            if d.delta_E is not None and d.delta_E < 10:
                strengths.append(f"good color match (ΔE={d.delta_E:.1f})")
            if d.spec.angle_independent:
                strengths.append("non-iridescent")
            if m.environmental_score and m.environmental_score.value > 0.8:
                strengths.append("eco-friendly")
            if m.scalability_score and m.scalability_score.value > 0.5:
                strengths.append("scalable")
            if d.delta_E is not None and d.delta_E > 30:
                weaknesses.append(f"poor color match (ΔE={d.delta_E:.0f})")
            if not d.spec.angle_independent:
                weaknesses.append("iridescent")

            all_rankings.append(MaterialRanking(
                material_class=d.material_class(),
                design_name=d.spec.name,
                design=d,
                metrics=m,
                composite_score=m.composite_score(),
                strengths=strengths,
                weaknesses=weaknesses,
            ))

    all_rankings.sort(key=lambda r: r.composite_score, reverse=True)

    # Best-in-category
    result = UnifiedDesignResult(
        target=target,
        rankings=all_rankings[:top_n],
        n_materials_evaluated=len(all_rankings),
    )

    if all_rankings:
        cap_sorted = sorted(all_rankings,
            key=lambda r: r.metrics.capacity_mg_g.value if r.metrics.capacity_mg_g else 0,
            reverse=True)
        result.best_for_capacity = cap_sorted[0].design_name if cap_sorted else ""

        cost_sorted = sorted(all_rankings,
            key=lambda r: r.metrics.cost_per_kg_usd.value if r.metrics.cost_per_kg_usd else 1e6)
        result.best_for_cost = cost_sorted[0].design_name if cost_sorted else ""

        sel_sorted = sorted(all_rankings,
            key=lambda r: r.metrics.selectivity_ratio.value if r.metrics.selectivity_ratio else 0,
            reverse=True)
        result.best_for_selectivity = sel_sorted[0].design_name if sel_sorted else ""

        scale_sorted = sorted(all_rankings,
            key=lambda r: r.metrics.scalability_score.value if r.metrics.scalability_score else 0,
            reverse=True)
        result.best_for_scalability = scale_sorted[0].design_name if scale_sorted else ""

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════

def print_material_report(result: UnifiedDesignResult, top_n: int = 10):
    """Print cross-material design comparison."""
    t = result.target
    print("=" * 85)
    print(f"MATERIAL DESIGN REPORT: {t.name}")
    print("=" * 85)
    print(f"\n  Target: {t.target_species} (d={t.target_diameter_A:.1f} Å, "
          f"q={t.target_charge:+d}, MW={t.target_mw:.1f})")
    print(f"  Conditions: pH={t.pH}, T={t.T_K:.0f} K, I={t.ionic_strength_M:.3f} M")
    if t.interferent_species:
        print(f"  Interferents: {', '.join(t.interferent_species)}")
    print(f"\n  Materials evaluated: {result.n_materials_evaluated}")

    print(f"\n{'Rank':>4} {'Material':<22} {'Class':<16} {'Cap':>7} {'Sel':>6} "
          f"{'t90':>6} {'$/kg':>7} {'Scale':>5} {'Score':>6}  Notes")
    print("-" * 105)

    for i, r in enumerate(result.rankings[:top_n]):
        m = r.metrics
        cap = f"{m.capacity_mg_g.value:.1f}" if m.capacity_mg_g else "  -"
        sel = f"{m.selectivity_ratio.value:.0f}" if m.selectivity_ratio else "  -"
        t90 = f"{m.kinetics_t90_min.value:.1f}" if m.kinetics_t90_min else "  -"
        cost = f"{m.cost_per_kg_usd.value:.0f}" if m.cost_per_kg_usd else "  -"
        scale = f"{m.scalability_score.value:.2f}" if m.scalability_score else "  -"
        notes = "; ".join(r.strengths[:2]) if r.strengths else ""
        if r.weaknesses:
            notes += f" [!{r.weaknesses[0]}]" if notes else f"[!{r.weaknesses[0]}]"

        print(f"{i+1:4d} {r.design_name:<22} {r.material_class:<16} "
              f"{cap:>7} {sel:>6} {t90:>6} {cost:>7} {scale:>5} "
              f"{r.composite_score:6.3f}  {notes}")

    print(f"\n  Best for capacity:     {result.best_for_capacity}")
    print(f"  Best for cost:         {result.best_for_cost}")
    print(f"  Best for selectivity:  {result.best_for_selectivity}")
    print(f"  Best for scalability:  {result.best_for_scalability}")

    # Tier distribution
    tiers = {1: 0, 2: 0, 3: 0}
    for r in result.rankings:
        m = r.metrics
        for field_name in ['capacity_mg_g', 'selectivity_ratio', 'kinetics_t90_min',
                           'cost_per_kg_usd', 'scalability_score']:
            tv = getattr(m, field_name, None)
            if tv is not None:
                tiers[tv.tier.value] += 1
    total = sum(tiers.values())
    if total > 0:
        print(f"\n  Data quality: T1={tiers[1]}/{total} T2={tiers[2]}/{total} T3={tiers[3]}/{total}")


if __name__ == "__main__":
    print("=" * 85)
    print("Material Designer — M1 + M5 Demo")
    print("=" * 85)

    # Selenite capture at Elk Valley conditions
    selenite = TargetSpec(
        name="Selenite Capture (Elk Valley)",
        target_type="oxoanion",
        target_species="SeO3^2-",
        target_diameter_A=3.5,
        target_charge=-2,
        target_mw=126.96,
        pH=5.0,
        T_K=283.15,  # ~10°C mountain water
        ionic_strength_M=0.01,
        interferent_species=["SO4^2-", "Ca^2+", "Mg^2+"],
    )

    result = unified_design(selenite, framework_func_groups=["amine-primary"])
    print_material_report(result)


# ═══════════════════════════════════════════════════════════════════════════
# M2: Polymeric Sorbents (Ion Exchange, Hydrogels, Functionalized Resins)
# ═══════════════════════════════════════════════════════════════════════════
#
# Physics:
#   Donnan equilibrium — ion partitioning into charged polymer phase
#   Flory-Rehner — swelling equilibrium for crosslinked networks
#   Ion exchange selectivity — from selectivity coefficients
#   Film/particle diffusion kinetics
#
# References:
#   Helfferich F. Ion Exchange. McGraw-Hill 1962 (reprinted Dover 1995).
#   Flory PJ. Principles of Polymer Chemistry. Cornell 1953.
#   Flory PJ, Rehner J. J. Chem. Phys. 1943, 11, 521.
#   Marcus Y. Ion Properties. CRC Press 1997.
#   Zagorodni AA. Ion Exchange Materials. Elsevier 2007.

# ── Published Polymer Sorbent Data (T1) ──────────────────────────────────

POLYMER_DATABASE = {
    # name: (type, capacity_meq_g, functional_group, matrix, selectivity_class, cost_class)
    # Capacity in meq/g (milliequivalents per gram dry resin)

    # Strong acid cation exchangers (SAC) — sulfonate groups, always ionized
    "Amberlite-IR120":  ("SAC", 4.4, "sulfonate", "PS-DVB", "cation", "low"),
    "Dowex-50WX8":      ("SAC", 5.0, "sulfonate", "PS-DVB", "cation", "low"),
    "Purolite-C100":    ("SAC", 4.2, "sulfonate", "PS-DVB", "cation", "low"),

    # Weak acid cation exchangers (WAC) — carboxylate groups, pH-dependent
    "Amberlite-IRC86":  ("WAC", 10.0, "carboxylate", "polyacrylic", "cation", "low"),
    "Dowex-MAC3":       ("WAC", 10.5, "carboxylate", "polyacrylic", "cation", "low"),

    # Strong base anion exchangers (SBA) — quaternary amine, always ionized
    "Amberlite-IRA400": ("SBA", 3.5, "quaternary-N", "PS-DVB", "anion", "moderate"),
    "Dowex-1X8":        ("SBA", 3.5, "quaternary-N", "PS-DVB", "anion", "moderate"),
    "Purolite-A300":    ("SBA", 3.8, "quaternary-N", "PS-DVB", "anion", "moderate"),

    # Weak base anion exchangers (WBA) — amine groups, pH-dependent
    "Amberlite-IRA67":  ("WBA", 5.5, "amine-primary", "polyacrylic", "anion", "low"),
    "Purolite-A100":    ("WBA", 5.0, "amine-primary", "PS-DVB", "anion", "low"),

    # Chelating resins — selective for heavy metals
    "Chelex-100":       ("chelating", 2.8, "iminodiacetate", "PS-DVB", "heavy-metal", "high"),
    "Lewatit-TP207":    ("chelating", 2.5, "iminodiacetate", "PS-DVB", "heavy-metal", "high"),
    "Duolite-GT73":     ("chelating", 1.3, "thiol", "PS-DVB", "soft-metal", "high"),

    # Specialty: amidoxime (uranium from seawater)
    "amidoxime-fiber":  ("specialty", 3.0, "amidoxime", "polyethylene", "uranium", "high"),

    # Hydrogels
    "polyHEMA":         ("hydrogel", 0.5, "hydroxyl", "crosslinked-HEMA", "general", "moderate"),
    "chitosan-bead":    ("hydrogel", 4.5, "amine-primary", "chitosan", "heavy-metal", "low"),
}

# Ion exchange selectivity coefficients (T1/T2)
# K_AB = preference of resin for ion A over ion B
# For SAC (sulfonate): selectivity series (Hofmeister-like)
# Ref: Helfferich 1962 Table 5.3; Marcus 1997
SAC_SELECTIVITY = {
    # ion: relative selectivity vs Na+ on sulfonate resin (8% DVB)
    "Li+": 0.85, "H+": 1.0, "Na+": 1.0, "K+": 1.7, "NH4+": 1.9,
    "Rb+": 2.0, "Cs+": 2.1, "Ag+": 4.0, "Tl+": 6.0,
    "Mg2+": 2.5, "Ca2+": 3.9, "Sr2+": 5.0, "Ba2+": 8.7,
    "Mn2+": 2.4, "Fe2+": 2.6, "Co2+": 2.8, "Ni2+": 2.9,
    "Cu2+": 2.7, "Zn2+": 2.6, "Cd2+": 2.9,
    "Pb2+": 5.0, "Hg2+": 7.0,
}

# For SBA (quaternary N): selectivity for anions
# Ref: Helfferich 1962; Zagorodni 2007
SBA_SELECTIVITY = {
    # ion: relative selectivity vs Cl- on Type I SBA
    "OH-": 0.06, "F-": 0.09, "Cl-": 1.0, "Br-": 2.8,
    "NO3-": 3.2, "I-": 8.7, "ClO4-": 10.0,
    "HCO3-": 0.32, "CH3COO-": 0.14,
    "SO4^2-": 0.15, "CrO4^2-": 1.0,
    "HPO4^2-": 0.25, "SeO3^2-": 0.50, "SeO4^2-": 0.17,
    "AsO4^3-": 0.10, "PO4^3-": 0.07,
}

# Polymer cost estimates
POLYMER_COST = {
    "low": TieredValue(15.0, DataTier.T1_KNOWN, "Bulk ion exchange resin: ~$15/kg", 20),
    "moderate": TieredValue(50.0, DataTier.T2_SOLID, "Specialty resin: ~$50/kg", 30),
    "high": TieredValue(200.0, DataTier.T3_CONCEPTUAL, "Chelating/specialty: ~$200/kg", 50),
}


# ── Donnan Equilibrium (T2) ──────────────────────────────────────────────

def donnan_partition(z_ion: int, C_ext_mM: float,
                      Q_resin_meq_mL: float,
                      z_co: int = 1) -> float:
    """Donnan equilibrium: ion concentration ratio inside/outside resin.

    For a resin with fixed charge Q and external electrolyte at concentration C,
    the co-ion exclusion and counter-ion enrichment follow:

    For monovalent counter-ion:
        C_in / C_ext = Q / C_ext  (approximate, high Q limit)

    General (ideal Donnan):
        (C_counter_in / C_counter_ext) = (C_co_ext / C_co_in)^(z_co/z_counter)

    Simplified for practical use:
        Enrichment factor ≈ (Q / C_ext)^(1/|z_ion|) for counter-ions
        Exclusion factor ≈ (C_ext / Q)^(|z_ion|) for co-ions

    Parameters
    ----------
    z_ion : int
        Charge of the ion (+1, +2, -1, -2, etc.).
    C_ext_mM : float
        External solution concentration (mM).
    Q_resin_meq_mL : float
        Resin capacity in meq per mL swollen resin.
    z_co : int
        Co-ion charge magnitude (default 1).

    Returns
    -------
    float
        Enrichment factor (C_in / C_ext). >1 = enriched, <1 = excluded.

    Physics tier: T2 (Donnan 1911, Helfferich 1962 Ch. 4).
    """
    if C_ext_mM <= 0 or Q_resin_meq_mL <= 0:
        return 1.0

    # Convert: Q in meq/mL ≈ mM equivalent (rough, assuming ~1 g/mL wet density)
    Q_mM = Q_resin_meq_mL * 1000.0  # meq/mL → μeq/mL ≈ mM

    ratio = Q_mM / C_ext_mM
    z_abs = abs(z_ion)

    if z_abs == 0:
        return 1.0  # neutral species, no Donnan effect

    # Counter-ion (sign of z opposite to resin charge → enriched)
    # We assume the resin charge sign is determined by the selectivity class
    # For simplicity: enrichment_factor = ratio^(1/|z|) for counter-ions
    return ratio ** (1.0 / z_abs)


def donnan_exclusion(z_co: int, C_ext_mM: float,
                      Q_resin_meq_mL: float) -> float:
    """Co-ion exclusion factor (Donnan).

    Co-ions (same charge as resin) are excluded from the resin phase.
    Exclusion factor = C_co_in / C_co_ext < 1.

    Physics tier: T2.
    """
    if C_ext_mM <= 0 or Q_resin_meq_mL <= 0:
        return 1.0

    Q_mM = Q_resin_meq_mL * 1000.0
    ratio = C_ext_mM / Q_mM
    z_abs = abs(z_co)
    return min(1.0, ratio ** z_abs)


# ── Ion Exchange Selectivity (T1/T2) ─────────────────────────────────────

def ion_exchange_selectivity(target_ion: str, interferent_ion: str,
                               resin_type: str = "SAC") -> float:
    """Selectivity coefficient for target vs interferent on a resin.

    α(A/B) = K_A / K_B where K values are from published selectivity tables.

    Parameters
    ----------
    target_ion : str
        Target ion (e.g., "Pb2+", "SeO3^2-").
    interferent_ion : str
        Competing ion (e.g., "Ca2+", "SO4^2-").
    resin_type : str
        "SAC" (sulfonate cation) or "SBA" (quaternary amine anion).

    Returns
    -------
    float
        Selectivity ratio (>1 = prefers target).

    Physics tier: T1 (published selectivity coefficients) / T3 (if estimated).
    """
    if resin_type == "SAC":
        table = SAC_SELECTIVITY
    elif resin_type == "SBA":
        table = SBA_SELECTIVITY
    else:
        return 1.0  # unknown resin type

    K_target = table.get(target_ion, 1.0)
    K_interf = table.get(interferent_ion, 1.0)

    if K_interf <= 0:
        return float('inf')
    return K_target / K_interf


# ── Flory-Rehner Swelling (T2) ───────────────────────────────────────────

def flory_rehner_swelling_ratio(chi: float, Mc: float, rho_p: float = 1.1,
                                  V1: float = 18.0) -> float:
    """Flory-Rehner equilibrium swelling ratio for crosslinked polymer.

    At swelling equilibrium, the elastic retraction (crosslinks) balances
    the osmotic drive (mixing entropy + ion osmotic pressure).

    Simplified Flory-Rehner for neutral gels:
        Q_vol = (V1 / (Mc × ν × (0.5 - χ)))^(3/5)

    where:
        V1 = molar volume of solvent (18 cm³/mol for water)
        Mc = molecular weight between crosslinks
        ν = specific volume of polymer ≈ 1/ρ_p
        χ = Flory-Huggins interaction parameter (0.3-0.5 for hydrophilic)

    Parameters
    ----------
    chi : float
        Flory-Huggins parameter. <0.5 = hydrophilic, >0.5 = hydrophobic.
    Mc : float
        Molecular weight between crosslinks (g/mol). Higher = more swelling.
    rho_p : float
        Dry polymer density (g/cm³).
    V1 : float
        Molar volume of solvent (cm³/mol). 18.0 for water.

    Returns
    -------
    float
        Volumetric swelling ratio Q = V_swollen / V_dry.

    Physics tier: T2 (Flory & Rehner 1943).
    """
    if chi >= 0.5:
        # Hydrophobic: minimal swelling
        return 1.1

    nu = 1.0 / rho_p  # specific volume cm³/g
    # Q ∝ (ν × Mc × (0.5 - χ) / V1)^(3/5)
    # Higher Mc = more swelling, lower chi = more hydrophilic = more swelling
    numerator = nu * Mc * (0.5 - chi)
    if numerator <= 0:
        return 1.0

    Q = (numerator / V1) ** 0.6
    return max(1.0, Q)


def hydrogel_water_content(Q_vol: float) -> float:
    """Water content from volumetric swelling ratio.

    wt% water = (Q - 1) / Q × 100

    Physics tier: T2.
    """
    if Q_vol <= 1.0:
        return 0.0
    return (Q_vol - 1.0) / Q_vol * 100.0


# ── Kinetics: Film and Particle Diffusion (T2) ───────────────────────────

def film_diffusion_t90(bead_radius_m: float, film_thickness_m: float = 50e-6,
                        D_film: float = 1e-9,
                        C_bulk_mM: float = 1.0,
                        Q_meq_g: float = 3.0,
                        rho_bead: float = 1.1e6) -> float:
    """Time to 90% equilibrium by film (external) diffusion.

    t_0.9 ≈ -ln(0.1) × R × Q × ρ / (3 × D_f × C_bulk / δ)

    Parameters
    ----------
    bead_radius_m : float
        Bead radius (m).
    film_thickness_m : float
        Nernst film thickness (m). Typical: 10-100 μm.
    D_film : float
        Diffusivity in the stagnant film (m²/s).
    C_bulk_mM : float
        Bulk solution concentration (mM).
    Q_meq_g : float
        Resin capacity (meq/g).
    rho_bead : float
        Wet bead density (g/m³). ~1.1e6 g/m³ = 1.1 g/cm³.

    Returns
    -------
    float
        t_90 in minutes.

    Physics tier: T2 (Boyd 1947, Helfferich 1962 Ch. 6).
    """
    if D_film <= 0 or C_bulk_mM <= 0:
        return 999.0

    C_bulk_mol_m3 = C_bulk_mM  # mM ≈ mol/m³ (1 mM = 1 mol/m³)
    Q_eq_m3 = Q_meq_g * rho_bead / 1000.0  # meq/m³ bead volume

    # Mass transfer coefficient: k_f = D / δ
    k_f = D_film / film_thickness_m

    # Characteristic time: τ = R × Q / (3 × k_f × C)
    tau = bead_radius_m * Q_eq_m3 / (3.0 * k_f * C_bulk_mol_m3)

    # t_90 ≈ 2.303 × τ (from -ln(0.1))
    t_90_s = 2.303 * tau
    return t_90_s / 60.0


def particle_diffusion_t90(bead_radius_m: float,
                             D_eff: float = 1e-11) -> float:
    """Time to 90% equilibrium by intraparticle diffusion.

    t_0.9 ≈ 0.307 × R² / D_eff

    (From Vermeulen's approximation for the particle diffusion model.)

    Parameters
    ----------
    bead_radius_m : float
        Bead radius (m).
    D_eff : float
        Effective intraparticle diffusivity (m²/s).
        Typical: 1e-11 to 1e-10 m²/s for ions in resin.

    Returns
    -------
    float
        t_90 in minutes.

    Physics tier: T2 (Vermeulen 1953, Helfferich 1962).
    """
    if D_eff <= 0:
        return 999.0
    t_90_s = 0.307 * bead_radius_m ** 2 / D_eff
    return t_90_s / 60.0


def rate_limiting_step(t_film_min: float, t_particle_min: float) -> str:
    """Identify rate-limiting step.

    The slower step controls the overall rate.
    """
    if t_film_min > t_particle_min * 2:
        return "film-diffusion"
    elif t_particle_min > t_film_min * 2:
        return "particle-diffusion"
    else:
        return "mixed"


# ── Polymer Sorbent Design Engine ─────────────────────────────────────────

@dataclass
class PolymericSorbentSpec:
    """Specification for a polymeric sorbent material."""
    resin_name: str = ""
    resin_type: str = ""          # SAC, WAC, SBA, WBA, chelating, hydrogel, specialty
    functional_group: str = ""
    matrix: str = ""              # PS-DVB, polyacrylic, chitosan, etc.
    selectivity_class: str = ""   # cation, anion, heavy-metal, soft-metal, etc.

    # Properties
    capacity_meq_g: float = 0.0   # total exchange capacity (meq/g dry)
    bead_diameter_mm: float = 0.5  # typical bead size
    water_content_pct: float = 50.0  # water content of swollen resin

    # Swelling (for hydrogels)
    chi_parameter: float = 0.4    # Flory-Huggins (lower = more hydrophilic)
    Mc_crosslink: float = 5000.0  # MW between crosslinks

    # Stability
    pH_range: Tuple[float, float] = (0.0, 14.0)
    max_temp_K: float = 393.15    # 120°C default
    regenerable: bool = True

    cost_class: str = "low"

    @classmethod
    def from_database(cls, name: str) -> 'PolymericSorbentSpec':
        if name not in POLYMER_DATABASE:
            raise KeyError(f"Unknown polymer '{name}'. "
                           f"Known: {sorted(POLYMER_DATABASE.keys())}")
        rtype, cap, fg, matrix, sel, cost = POLYMER_DATABASE[name]
        return cls(
            resin_name=name, resin_type=rtype,
            functional_group=fg, matrix=matrix,
            selectivity_class=sel, capacity_meq_g=cap,
            cost_class=cost,
        )


@dataclass
class PolymericSorbentDesign(MaterialDesign):
    """Complete polymeric sorbent design."""
    spec: PolymericSorbentSpec = field(default_factory=PolymericSorbentSpec)
    target: TargetSpec = field(default_factory=TargetSpec)
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)

    # Design details
    q_max_mg_g: float = 0.0
    donnan_enrichment: float = 1.0
    selectivity_vs_worst: float = 1.0
    t_film_min: float = 0.0
    t_particle_min: float = 0.0
    rate_limiting: str = ""
    func_group_activity: float = 1.0
    swelling_ratio: float = 1.0

    def material_class(self) -> str:
        return f"PolymericSorbent ({self.spec.resin_type})"

    def design_summary(self) -> str:
        lines = [
            f"Resin: {self.spec.resin_name} ({self.spec.resin_type})",
            f"Functional group: {self.spec.functional_group} on {self.spec.matrix}",
            f"Capacity: {self.spec.capacity_meq_g:.1f} meq/g",
        ]
        if self.metrics.capacity_mg_g:
            lines.append(f"Predicted capacity: {self.metrics.capacity_mg_g}")
        return "\n".join(lines)

    def predict_performance(self, target: TargetSpec = None) -> PerformanceMetrics:
        if target is None:
            target = self.target
        self.target = target
        s = self.spec

        # 1. Functional group activity at pH
        is_anion = target.target_charge < 0
        self.func_group_activity = func_group_fraction_active(
            s.functional_group, target.pH,
            for_anion_capture=is_anion
        )

        # 2. Effective capacity
        effective_cap_meq = s.capacity_meq_g * self.func_group_activity
        mw = target.target_mw if target.target_mw > 0 else 80.0
        z_abs = max(1, abs(target.target_charge))
        # meq/g to mg/g: meq × (MW / |z|) = mg
        self.q_max_mg_g = effective_cap_meq * mw / z_abs

        # 3. Donnan enrichment
        Q_meq_mL = s.capacity_meq_g * 0.5  # rough: 0.5 g dry resin per mL wet
        C_ext = 0.1  # mM default trace concentration
        if target.target_charge != 0:
            self.donnan_enrichment = donnan_partition(
                target.target_charge, C_ext, Q_meq_mL
            )

        # 4. Selectivity
        if s.resin_type == "SAC" and target.target_charge > 0:
            sel_table_type = "SAC"
        elif s.resin_type in ("SBA", "WBA") and target.target_charge < 0:
            sel_table_type = "SBA"
        else:
            sel_table_type = None

        worst_selectivity = 1.0
        sel_tier = DataTier.T2_SOLID
        sel_source = "repulsion physics"

        # Priority 1: Published IX selectivity coefficients (T1)
        if sel_table_type and target.interferent_species:
            sels = []
            for intf in target.interferent_species:
                intf_clean = intf.replace("^", "")
                target_clean = target.target_species.replace("^", "")
                sel = ion_exchange_selectivity(target_clean, intf_clean, sel_table_type)
                sels.append(sel)
            if sels:
                worst_selectivity = min(sels)
                sel_tier = DataTier.T1_KNOWN
                sel_source = f"IX selectivity table ({sel_table_type})"

        # Priority 2: Repulsion physics (T2) for chelating/specialty resins
        elif target.interferent_species:
            try:
                site_spec = _polymer_to_site_spec(s)
                worst_selectivity = _compute_repulsion_selectivity(
                    site_spec, target.target_species,
                    target.interferent_species,
                )
                worst_selectivity = max(0.01, min(1e6, worst_selectivity))
                sel_source = "differential repulsion (R1-R5)"
            except (ImportError, Exception):
                worst_selectivity = 1.0
                sel_tier = DataTier.T3_CONCEPTUAL
                sel_source = "no selectivity data"

        self.selectivity_vs_worst = worst_selectivity

        # 5. Kinetics
        R_bead = s.bead_diameter_mm * 1e-3 / 2.0  # m
        self.t_film_min = film_diffusion_t90(R_bead)
        self.t_particle_min = particle_diffusion_t90(R_bead)
        self.rate_limiting = rate_limiting_step(self.t_film_min, self.t_particle_min)
        t90 = max(self.t_film_min, self.t_particle_min)

        # 6. Swelling (hydrogels)
        if s.resin_type == "hydrogel":
            self.swelling_ratio = flory_rehner_swelling_ratio(
                s.chi_parameter, s.Mc_crosslink
            )

        # 7. Langmuir equilibrium capacity at trace concentration
        dG_bind = -20.0  # default moderate
        if s.resin_type == "chelating":
            dG_bind = -35.0
        elif s.selectivity_class == "soft-metal":
            dG_bind = -40.0
        K_L = langmuir_K_from_dG(dG_bind, target.T_K)

        # Ionic strength correction (T2: Davies)
        I = target.ionic_strength_M
        if I > 0 and target.target_charge != 0:
            K_L = correct_K_for_ionic_strength(
                K_L, target.target_charge, 0, I
            )

        # Single-component capacity
        C_eq = 0.1
        q_eq_single = langmuir_capacity(self.q_max_mg_g, K_L, C_eq)

        # Competitive Langmuir
        q_eq = q_eq_single
        if target.interferent_species and target.interferent_concentrations_mM:
            K_intfs = []
            C_intfs = list(target.interferent_concentrations_mM)
            for intf in target.interferent_species:
                K_i = K_L * 0.3  # interferents bind weaker on IX resins
                K_intfs.append(K_i)
            while len(C_intfs) < len(K_intfs):
                C_intfs.append(1.0)
            q_eq = competitive_langmuir(
                self.q_max_mg_g, K_L, C_eq, K_intfs, C_intfs
            )

        # 8. Cost
        cost = POLYMER_COST.get(s.cost_class, POLYMER_COST["moderate"])

        # 9. Scalability — ion exchange is industrially mature
        scale_map = {
            "SAC": 0.95, "SBA": 0.95, "WAC": 0.90, "WBA": 0.90,
            "chelating": 0.75, "specialty": 0.40, "hydrogel": 0.50,
        }
        scalability = scale_map.get(s.resin_type, 0.5)

        # 10. Environmental
        env_map = {
            "SAC": 0.6, "SBA": 0.5, "WAC": 0.7, "WBA": 0.7,
            "chelating": 0.5, "hydrogel": 0.7, "specialty": 0.4,
        }
        env = env_map.get(s.resin_type, 0.5)
        # Chitosan is bio-based → bonus
        if s.matrix == "chitosan":
            env = min(1.0, env + 0.2)

        self.metrics = PerformanceMetrics(
            capacity_mg_g=TieredValue(q_eq, DataTier.T2_SOLID,
                f"Langmuir + IX capacity: q_max={self.q_max_mg_g:.1f} mg/g", 30),
            selectivity_ratio=TieredValue(worst_selectivity, sel_tier,
                sel_source, 20 if sel_tier == DataTier.T1_KNOWN else 30),
            kinetics_t90_min=TieredValue(t90, DataTier.T2_SOLID,
                f"Film+particle diffusion, {self.rate_limiting}", 30),
            cost_per_kg_usd=cost,
            scalability_score=TieredValue(scalability, DataTier.T2_SOLID,
                f"{s.resin_type} industrial maturity"),
            durability_cycles=TieredValue(
                500.0 if s.regenerable else 1.0, DataTier.T2_SOLID,
                "IX resins: ~500 regeneration cycles typical"),
            environmental_score=TieredValue(env, DataTier.T3_CONCEPTUAL,
                f"{s.matrix} environmental estimate"),
        )

        return self.metrics


def design_polymer(resin_name: str, target: TargetSpec) -> PolymericSorbentDesign:
    """Design a polymeric sorbent for a target."""
    spec = PolymericSorbentSpec.from_database(resin_name)
    design = PolymericSorbentDesign(spec=spec)
    design.predict_performance(target)
    return design


def screen_polymers(target: TargetSpec, top_n: int = 10) -> List[PolymericSorbentDesign]:
    """Screen all known polymeric sorbents against a target."""
    designs = []
    for name in POLYMER_DATABASE:
        try:
            d = design_polymer(name, target)
            designs.append(d)
        except Exception:
            pass
    designs.sort(key=lambda d: d.metrics.composite_score(), reverse=True)
    return designs[:top_n]


# ═══════════════════════════════════════════════════════════════════════════
# M3: Composite Materials (Scaffold-on-Support, Core-Shell, Packed Beds)
# ═══════════════════════════════════════════════════════════════════════════
#
# Physics (all T2 — established engineering equations):
#   Maxwell-Garnett effective medium theory (1904)
#   Bruggeman symmetric effective medium (1935)
#   Rule of mixtures (linear, Voigt/Reuss bounds)
#   Kozeny-Carman pressure drop (1937)
#   Shell diffusion for core-shell geometry
#
# Data policy: T1 (published) or T2 (textbook equations) ONLY.
# No T3 estimates. Missing data returned as None, not fabricated.
#
# References:
#   Maxwell Garnett JC. Phil. Trans. R. Soc. 1904, 203, 385.
#   Bruggeman DAG. Ann. Phys. 1935, 416, 636.
#   Kozeny J. Sitzber. Akad. Wiss. Wien 1927, 136, 271.
#   Carman PC. Trans. Inst. Chem. Eng. 1937, 15, 150.
#   Perry RH, Green DW. Perry's Chemical Engineers' Handbook, 8th ed., McGraw-Hill 2008.

# ── Effective Medium Theory (T2) ─────────────────────────────────────────

def maxwell_garnett(prop_matrix: float, prop_inclusion: float,
                     volume_fraction: float) -> float:
    """Maxwell-Garnett effective medium: dilute inclusions in a matrix.

    For a property P (dielectric, thermal conductivity, etc.):
    P_eff = P_m × (P_i + 2P_m + 2f(P_i - P_m)) / (P_i + 2P_m - f(P_i - P_m))

    Valid for f < ~0.3 (dilute limit). Spherical inclusions assumed.

    Parameters
    ----------
    prop_matrix : float
        Property of the continuous matrix phase.
    prop_inclusion : float
        Property of the inclusion (dispersed) phase.
    volume_fraction : float
        Volume fraction of inclusions (0–1).

    Returns
    -------
    float
        Effective composite property.

    Physics tier: T2 (Maxwell Garnett 1904).
    """
    Pm, Pi, f = prop_matrix, prop_inclusion, volume_fraction
    if f <= 0:
        return Pm
    if f >= 1:
        return Pi
    num = Pi + 2*Pm + 2*f*(Pi - Pm)
    den = Pi + 2*Pm - f*(Pi - Pm)
    if den == 0:
        return Pm
    return Pm * num / den


def bruggeman_ema(prop_a: float, prop_b: float,
                   f_a: float, tol: float = 1e-8,
                   max_iter: int = 100) -> float:
    """Bruggeman symmetric effective medium approximation.

    Self-consistent equation (no distinguished matrix):
    f_a × (P_a - P_eff)/(P_a + 2P_eff) + f_b × (P_b - P_eff)/(P_b + 2P_eff) = 0

    Solved iteratively. Valid for any volume fraction.

    Parameters
    ----------
    prop_a, prop_b : float
        Properties of components A and B.
    f_a : float
        Volume fraction of component A (f_b = 1 - f_a).

    Returns
    -------
    float
        Effective composite property.

    Physics tier: T2 (Bruggeman 1935).
    """
    f_b = 1.0 - f_a
    # Initial guess: rule of mixtures
    P_eff = f_a * prop_a + f_b * prop_b

    for _ in range(max_iter):
        if P_eff <= 0:
            P_eff = 0.5 * (prop_a + prop_b)
        term_a = f_a * (prop_a - P_eff) / (prop_a + 2*P_eff) if (prop_a + 2*P_eff) != 0 else 0
        term_b = f_b * (prop_b - P_eff) / (prop_b + 2*P_eff) if (prop_b + 2*P_eff) != 0 else 0
        residual = term_a + term_b

        # Newton step: d(residual)/d(P_eff)
        dra = -f_a * (prop_a + 2*P_eff + 2*(prop_a - P_eff)) / (prop_a + 2*P_eff)**2 if (prop_a + 2*P_eff) != 0 else 0
        drb = -f_b * (prop_b + 2*P_eff + 2*(prop_b - P_eff)) / (prop_b + 2*P_eff)**2 if (prop_b + 2*P_eff) != 0 else 0
        deriv = dra + drb

        if abs(deriv) < 1e-30:
            break
        P_eff -= residual / deriv

        if abs(residual) < tol:
            break

    return max(0.0, P_eff)


# ── Rule of Mixtures (T2) ────────────────────────────────────────────────

def rule_of_mixtures(prop_a: float, prop_b: float, f_a: float) -> float:
    """Linear rule of mixtures (Voigt upper bound).

    P_eff = f_a × P_a + (1 - f_a) × P_b

    Exact for additive properties (density, cost per volume).
    Upper bound for transport properties.

    Physics tier: T2 (thermodynamic mixing).
    """
    return f_a * prop_a + (1.0 - f_a) * prop_b


def inverse_rule_of_mixtures(prop_a: float, prop_b: float, f_a: float) -> float:
    """Inverse (Reuss) rule of mixtures — lower bound.

    1/P_eff = f_a/P_a + (1-f_a)/P_b

    Lower bound for transport properties (series arrangement).

    Physics tier: T2.
    """
    if prop_a <= 0 or prop_b <= 0:
        return 0.0
    f_b = 1.0 - f_a
    return 1.0 / (f_a / prop_a + f_b / prop_b)


# ── Composite Capacity (T2) ──────────────────────────────────────────────

def composite_capacity(q_active_mg_g: float, f_active_mass: float) -> float:
    """Capacity of a composite = active fraction × active material capacity.

    q_composite = f_mass_active × q_active

    This is exact (mass balance). The support contributes zero capacity.

    Parameters
    ----------
    q_active_mg_g : float
        Capacity of the active sorbent alone (mg/g).
    f_active_mass : float
        Mass fraction of active sorbent in the composite (0–1).

    Returns
    -------
    float
        Composite capacity (mg/g composite).

    Physics tier: T2 (mass balance, exact).
    """
    return q_active_mg_g * f_active_mass


def composite_density(rho_active: float, rho_support: float,
                       f_vol_active: float) -> float:
    """Composite density from volume-weighted components.

    ρ_composite = f_vol × ρ_active + (1 - f_vol) × ρ_support

    Physics tier: T2 (exact for non-porous composites).
    """
    return rule_of_mixtures(rho_active, rho_support, f_vol_active)


def composite_cost(cost_active_per_kg: float, cost_support_per_kg: float,
                    f_mass_active: float) -> float:
    """Composite cost from mass-weighted components.

    $/kg_composite = f_mass × $/kg_active + (1-f_mass) × $/kg_support

    Physics tier: T2 (exact, accounting).
    """
    return rule_of_mixtures(cost_active_per_kg, cost_support_per_kg, f_mass_active)


# ── Core-Shell Geometry (T2) ─────────────────────────────────────────────

def core_shell_volume_fraction(R_core_m: float, shell_thickness_m: float) -> float:
    """Volume fraction of shell in a core-shell particle.

    f_shell = 1 - (R_core / R_total)³

    Physics tier: T2 (geometry, exact).
    """
    R_total = R_core_m + shell_thickness_m
    if R_total <= 0:
        return 0.0
    return 1.0 - (R_core_m / R_total) ** 3


def core_shell_mass_fraction(R_core_m: float, shell_thickness_m: float,
                               rho_core: float, rho_shell: float) -> float:
    """Mass fraction of shell material in a core-shell particle.

    f_mass_shell = f_vol_shell × ρ_shell / ρ_composite

    Physics tier: T2 (geometry + density, exact).
    """
    f_vol = core_shell_volume_fraction(R_core_m, shell_thickness_m)
    rho_comp = composite_density(rho_shell, rho_core, f_vol)
    if rho_comp <= 0:
        return 0.0
    return f_vol * rho_shell / rho_comp


def shell_diffusion_t90(R_total_m: float, R_core_m: float,
                          D_shell: float = 1e-11) -> float:
    """Diffusion time through a shell layer.

    Approximate: t_90 ≈ 0.307 × (R_total - R_core)² / D_shell

    For core-shell particles, only the shell is active. Diffusion
    path length is shell thickness, not full particle radius.

    Parameters
    ----------
    R_total_m : float
        Total particle radius (m).
    R_core_m : float
        Inert core radius (m).
    D_shell : float
        Effective diffusivity in shell (m²/s).

    Returns
    -------
    float
        t_90 in minutes.

    Physics tier: T2 (Fick's law in spherical shell geometry).
    """
    shell_thickness = R_total_m - R_core_m
    if shell_thickness <= 0 or D_shell <= 0:
        return 0.0
    t_90_s = 0.307 * shell_thickness ** 2 / D_shell
    return t_90_s / 60.0


# ── Kozeny-Carman Pressure Drop (T2) ─────────────────────────────────────

def kozeny_carman_dP(v_superficial_m_s: float, bed_length_m: float,
                      dp_m: float, porosity: float = 0.4,
                      viscosity: float = 8.9e-4) -> float:
    """Kozeny-Carman equation: pressure drop across packed bed.

    ΔP/L = 150 × η × v × (1-ε)² / (ε³ × dp²)

    Valid for laminar flow (Re_p < 10, typical for IX/sorbent beds).

    Parameters
    ----------
    v_superficial_m_s : float
        Superficial velocity (m/s).
    bed_length_m : float
        Bed length (m).
    dp_m : float
        Particle diameter (m).
    porosity : float
        Bed void fraction (0–1). Typical: 0.35–0.45 for packed spheres.
    viscosity : float
        Dynamic viscosity (Pa·s).

    Returns
    -------
    float
        Pressure drop in Pa.

    Physics tier: T2 (Kozeny 1927, Carman 1937).
    """
    if dp_m <= 0 or porosity <= 0 or porosity >= 1:
        return 0.0
    dP_per_L = 150.0 * viscosity * v_superficial_m_s * (1 - porosity)**2 / \
               (porosity**3 * dp_m**2)
    return dP_per_L * bed_length_m


def kozeny_carman_dP_kPa(v_superficial_m_s: float, bed_length_m: float,
                           dp_m: float, porosity: float = 0.4,
                           viscosity: float = 8.9e-4) -> float:
    """Kozeny-Carman pressure drop in kPa."""
    return kozeny_carman_dP(v_superficial_m_s, bed_length_m, dp_m,
                             porosity, viscosity) / 1000.0


# ── Breakthrough Estimation (T2) ─────────────────────────────────────────

def bed_volumes_to_breakthrough(q_mg_g: float, rho_bed_kg_m3: float,
                                  porosity: float, C_feed_mg_L: float) -> float:
    """Estimate bed volumes (BV) to breakthrough.

    BV = q × ρ_bed × (1 - ε) / C_feed

    Assumes ideal plug flow with sharp front (Thomas model limit).
    Real breakthrough curves are broader due to dispersion + kinetics.

    Parameters
    ----------
    q_mg_g : float
        Sorbent capacity at feed concentration (mg/g).
    rho_bed_kg_m3 : float
        Packed bed density (kg/m³). Typical: 600-900 for IX resins.
    porosity : float
        Bed void fraction.
    C_feed_mg_L : float
        Feed concentration (mg/L).

    Returns
    -------
    float
        Bed volumes to breakthrough (dimensionless).

    Physics tier: T2 (mass balance, ideal).
    """
    if C_feed_mg_L <= 0:
        return float('inf')
    # q in mg/g → mg/kg × ρ_bed × (1-ε) / C_feed
    q_mg_kg = q_mg_g * 1000.0  # mg/g → mg/kg
    return q_mg_kg * rho_bed_kg_m3 * (1.0 - porosity) / (C_feed_mg_L * 1000.0)
    # The 1000 converts L to m³: C_feed mg/L × 1000 L/m³ = mg/m³


# ── Composite Configurations Database (T1 data only) ─────────────────────

@dataclass
class CompositeConfig:
    """A published composite material configuration.

    All properties from published data (T1). No estimates.
    """
    name: str
    description: str
    active_material: str        # what provides the sorption
    support_material: str       # structural support / carrier
    configuration: str          # "core-shell", "coating", "impregnated", "blended"

    # Published properties (None = not available, do NOT estimate)
    active_fraction_mass: Optional[float] = None     # mass fraction of active
    particle_diameter_mm: Optional[float] = None      # typical particle size
    density_kg_m3: Optional[float] = None             # bulk packed density
    published_capacity_mg_g: Optional[float] = None   # published capacity for a specific target
    published_target: str = ""                         # what the capacity was measured for
    published_selectivity: Optional[float] = None
    published_BV_breakthrough: Optional[float] = None

    # References (DOI or specific citation)
    source: str = ""


# Published composite systems — T1 data only, no estimates
COMPOSITE_DATABASE: Dict[str, CompositeConfig] = {}


def _add_composite(c: CompositeConfig):
    COMPOSITE_DATABASE[c.name] = c


_add_composite(CompositeConfig(
    name="FeOOH-on-sand",
    description="Iron oxyhydroxide coated sand for arsenic/selenite removal",
    active_material="FeOOH (goethite/ferrihydrite)",
    support_material="silica sand",
    configuration="coating",
    active_fraction_mass=0.05,
    particle_diameter_mm=0.5,
    density_kg_m3=1500.0,
    published_capacity_mg_g=1.5,
    published_target="As(V)",
    published_BV_breakthrough=5000.0,
    source="Thirunavukkarasu OS et al. Water Res. 2003, 37, 4500. DOI:10.1016/S0043-1354(03)00395-4",
))

_add_composite(CompositeConfig(
    name="GAC-FeOx",
    description="Granular activated carbon impregnated with iron oxide",
    active_material="Fe₂O₃ / FeOOH",
    support_material="granular activated carbon",
    configuration="impregnated",
    active_fraction_mass=0.10,
    particle_diameter_mm=1.0,
    density_kg_m3=500.0,
    published_capacity_mg_g=4.7,
    published_target="As(V)",
    published_BV_breakthrough=10000.0,
    source="Gu Z, Fang J, Deng B. Environ. Sci. Technol. 2005, 39, 3833. DOI:10.1021/es048179r",
))

_add_composite(CompositeConfig(
    name="TiO2-on-alumina",
    description="TiO₂ coating on alumina beads for selenate photoreduction",
    active_material="TiO₂ (anatase)",
    support_material="γ-Al₂O₃ beads",
    configuration="coating",
    active_fraction_mass=0.08,
    particle_diameter_mm=3.0,
    density_kg_m3=1200.0,
    published_capacity_mg_g=None,  # photocatalytic, not adsorptive
    published_target="Se(VI) photoreduction",
    source="Zhang N et al. J. Hazard. Mater. 2008, 150, 481. DOI:10.1016/j.jhazmat.2007.04.131",
))

_add_composite(CompositeConfig(
    name="MOF-on-fiber",
    description="UiO-66-NH₂ grown on electrospun PAN fiber",
    active_material="UiO-66-NH₂",
    support_material="PAN nanofiber",
    configuration="coating",
    active_fraction_mass=0.30,
    particle_diameter_mm=None,  # fiber mat, not beads
    density_kg_m3=None,
    published_capacity_mg_g=62.0,
    published_target="Cr(VI)",
    source="Efome JE et al. Chem. Eng. J. 2018, 352, 737. DOI:10.1016/j.cej.2018.07.035",
))

_add_composite(CompositeConfig(
    name="chitosan-MMT",
    description="Chitosan-montmorillonite composite beads",
    active_material="chitosan + montmorillonite clay",
    support_material="chitosan matrix",
    configuration="blended",
    active_fraction_mass=0.60,
    particle_diameter_mm=2.0,
    density_kg_m3=800.0,
    published_capacity_mg_g=35.7,
    published_target="Cr(VI)",
    published_BV_breakthrough=None,
    source="Futalan CM et al. Carbohydr. Polym. 2011, 83, 528. DOI:10.1016/j.carbpol.2010.08.024",
))

_add_composite(CompositeConfig(
    name="zeolite-cement",
    description="Natural zeolite blended with Portland cement",
    active_material="clinoptilolite",
    support_material="Portland cement",
    configuration="blended",
    active_fraction_mass=0.50,
    particle_diameter_mm=5.0,
    density_kg_m3=1800.0,
    published_capacity_mg_g=11.0,
    published_target="NH₄⁺",
    source="Widiastuti N et al. Desalination 2011, 277, 15. DOI:10.1016/j.desal.2011.03.030",
))

_add_composite(CompositeConfig(
    name="MnO2-on-sand",
    description="MnO₂-coated sand for radium and heavy metal removal",
    active_material="MnO₂ (birnessite)",
    support_material="silica sand",
    configuration="coating",
    active_fraction_mass=0.03,
    particle_diameter_mm=0.7,
    density_kg_m3=1550.0,
    published_capacity_mg_g=0.8,
    published_target="Ra²⁺",
    published_BV_breakthrough=20000.0,
    source="Qian J et al. Water Res. 2019, 148, 49. DOI:10.1016/j.watres.2018.10.027",
))

_add_composite(CompositeConfig(
    name="alumina-coated-sand",
    description="Activated alumina coated on sand for fluoride removal",
    active_material="activated alumina (γ-Al₂O₃)",
    support_material="silica sand",
    configuration="coating",
    active_fraction_mass=0.07,
    particle_diameter_mm=0.6,
    density_kg_m3=1500.0,
    published_capacity_mg_g=3.5,
    published_target="F⁻",
    published_BV_breakthrough=8000.0,
    source="Tripathy SS, Raichur AM. J. Hazard. Mater. 2008, 153, 1043. DOI:10.1016/j.jhazmat.2007.09.100",
))


# ── Composite Design Engine ──────────────────────────────────────────────

@dataclass
class CompositeDesign(MaterialDesign):
    """Complete composite material design."""
    config: CompositeConfig = field(default_factory=lambda: CompositeConfig(
        name="", description="", active_material="", support_material="",
        configuration=""))
    target: TargetSpec = field(default_factory=TargetSpec)
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)

    # Computed from physics
    effective_capacity_mg_g: Optional[float] = None
    pressure_drop_kPa: Optional[float] = None
    BV_to_breakthrough: Optional[float] = None

    def material_class(self) -> str:
        return f"Composite ({self.config.configuration})"

    def design_summary(self) -> str:
        c = self.config
        lines = [
            f"Composite: {c.name}",
            f"  Active: {c.active_material} on {c.support_material} ({c.configuration})",
        ]
        if c.active_fraction_mass is not None:
            lines.append(f"  Active fraction: {c.active_fraction_mass:.0%}")
        if c.published_capacity_mg_g is not None:
            lines.append(f"  Published capacity: {c.published_capacity_mg_g} mg/g "
                         f"({c.published_target})")
        return "\n".join(lines)

    def predict_performance(self, target: TargetSpec = None) -> PerformanceMetrics:
        if target is None:
            target = self.target
        self.target = target
        c = self.config

        # Capacity: use published value if target matches, else scale from active fraction
        capacity_val = None
        capacity_tier = DataTier.T1_KNOWN
        capacity_source = ""

        if c.published_capacity_mg_g is not None:
            capacity_val = c.published_capacity_mg_g
            capacity_source = f"Published: {c.published_capacity_mg_g} mg/g for {c.published_target}"
            # If target doesn't match published target, note this
            if c.published_target and target.target_species and \
               target.target_species not in c.published_target:
                capacity_source += f" (measured for {c.published_target}, not {target.target_species})"
                capacity_tier = DataTier.T2_SOLID  # cross-applied, not direct measurement

        # Pressure drop (T2, if bed parameters available)
        self.pressure_drop_kPa = None
        if c.particle_diameter_mm is not None and c.particle_diameter_mm > 0:
            # Standard conditions: v=1e-3 m/s, L=1m, ε=0.4
            v_std = 1e-3  # m/s
            L_std = 1.0   # m
            dp_m = c.particle_diameter_mm * 1e-3
            self.pressure_drop_kPa = kozeny_carman_dP_kPa(v_std, L_std, dp_m)

        # Bed volumes to breakthrough (T2 from mass balance, if data available)
        self.BV_to_breakthrough = None
        if c.published_BV_breakthrough is not None:
            self.BV_to_breakthrough = c.published_BV_breakthrough
        elif (capacity_val is not None and c.density_kg_m3 is not None
              and capacity_val > 0):
            # Estimate from capacity + bed density
            C_feed_mg_L = 0.1  # 100 μg/L typical trace contaminant
            self.BV_to_breakthrough = bed_volumes_to_breakthrough(
                capacity_val, c.density_kg_m3, 0.4, C_feed_mg_L
            )

        # Kinetics: shell diffusion if core-shell
        kinetics_val = None
        kinetics_source = ""
        if c.configuration in ("core-shell", "coating") and c.particle_diameter_mm is not None:
            R_total = c.particle_diameter_mm * 1e-3 / 2.0
            f_active = c.active_fraction_mass if c.active_fraction_mass else 0.1
            # Shell thickness from volume fraction
            R_core = R_total * (1.0 - f_active) ** (1.0/3.0)
            kinetics_val = shell_diffusion_t90(R_total, R_core)
            kinetics_source = "Shell diffusion (T2)"

        # Selectivity: only report if published
        sel_val = None
        sel_tier = DataTier.T1_KNOWN
        if c.published_selectivity is not None:
            sel_val = c.published_selectivity

        # Cost: only from components if fractions are known
        cost_val = None
        # Not estimating — would require T3 component costs. Leave as None.

        # Scalability: composites are generally scalable (simple manufacturing)
        # Based on configuration type — this is T2 (engineering knowledge)
        scale_map = {
            "coating": 0.85,      # coating processes are industrial
            "impregnated": 0.80,  # impregnation is straightforward
            "blended": 0.75,      # mixing/extrusion is standard
            "core-shell": 0.60,   # more complex manufacturing
        }
        scalability = scale_map.get(c.configuration, 0.7)

        self.metrics = PerformanceMetrics(
            capacity_mg_g=TieredValue(capacity_val, capacity_tier, capacity_source)
                if capacity_val is not None else None,
            selectivity_ratio=TieredValue(sel_val, sel_tier, "Published")
                if sel_val is not None else None,
            kinetics_t90_min=TieredValue(kinetics_val, DataTier.T2_SOLID, kinetics_source)
                if kinetics_val is not None else None,
            cost_per_kg_usd=None,  # not estimating without published data
            scalability_score=TieredValue(scalability, DataTier.T2_SOLID,
                f"{c.configuration} process scalability"),
            durability_cycles=None,  # not estimating
            environmental_score=None,  # not estimating
        )

        return self.metrics


def design_composite(config_name: str, target: TargetSpec) -> CompositeDesign:
    """Design a composite material for a target."""
    if config_name not in COMPOSITE_DATABASE:
        raise KeyError(f"Unknown composite '{config_name}'. "
                       f"Known: {sorted(COMPOSITE_DATABASE.keys())}")
    config = COMPOSITE_DATABASE[config_name]
    design = CompositeDesign(config=config)
    design.predict_performance(target)
    return design


def screen_composites(target: TargetSpec, top_n: int = 10) -> List[CompositeDesign]:
    """Screen all known composites against a target."""
    designs = []
    for name in COMPOSITE_DATABASE:
        try:
            d = design_composite(name, target)
            designs.append(d)
        except Exception:
            pass
    designs.sort(key=lambda d: d.metrics.composite_score(), reverse=True)
    return designs[:top_n]


# ═══════════════════════════════════════════════════════════════════════════
# M4: Structural Color Materials (Bridge to Optical Pipeline)
# ═══════════════════════════════════════════════════════════════════════════
#
# Bridges MABE's optical pipeline (optical/) into the unified material
# design framework. Maps color targets to physical designs.
#
# Forward models (from optical/):
#   Bragg opal: λ_peak = 1.633 × D × n_eff (Module 2)
#   Photonic glass: Mie × Percus-Yevick (Module 6) — requires miepython
#   TMM multilayer: transfer matrix method (Module 7)
#   CIE color: spectrum → XYZ → Lab → sRGB (Module 9)
#
# All physics T2 (exact EM theory). Material data T1 (published n(λ)).
#
# References:
#   Bragg WH, Bragg WL. Proc. R. Soc. 1913, 88, 428.
#   Mie G. Ann. Phys. 1908, 330, 377.
#   Yeh P. Optical Waves in Layered Media. Wiley 2005.
#   CIE. Colorimetry, 4th ed. CIE 015:2018.

# ── Structural Color Database (T1: published) ────────────────────────────

@dataclass
class StructuralColorEntry:
    """Published structural color system."""
    name: str
    approach: str           # "bragg_opal", "photonic_glass", "multilayer", "BCP", "CNC"
    sphere_material: str    # e.g., "SiO2", "polystyrene"
    matrix_material: str    # e.g., "air", "PDMS", "water"
    diameter_nm: Optional[float] = None        # particle diameter
    packing_fraction: Optional[float] = None
    absorber: str = ""                          # e.g., "carbon_black", "melanin"
    absorber_fraction: Optional[float] = None
    angle_independent: bool = False             # non-iridescent?
    published_color: str = ""                   # e.g., "blue", "green"
    published_peak_nm: Optional[float] = None
    scalability: str = ""     # "lab", "pilot", "industrial"
    source: str = ""


STRUCTURAL_COLOR_DB: Dict[str, StructuralColorEntry] = {}


def _add_sc(entry: StructuralColorEntry):
    STRUCTURAL_COLOR_DB[entry.name] = entry


_add_sc(StructuralColorEntry(
    name="SiO2-opal-blue",
    approach="bragg_opal",
    sphere_material="SiO2",
    matrix_material="air",
    diameter_nm=207.0,
    packing_fraction=0.74,
    angle_independent=False,
    published_color="blue",
    published_peak_nm=450.0,
    scalability="lab",
    source="Gao W et al. J. Nanopart. Res. 2017, 19, 37. DOI:10.1007/s11051-017-3735-7",
))

_add_sc(StructuralColorEntry(
    name="SiO2-opal-green",
    approach="bragg_opal",
    sphere_material="SiO2",
    matrix_material="air",
    diameter_nm=260.0,
    packing_fraction=0.74,
    angle_independent=False,
    published_color="green",
    published_peak_nm=530.0,
    scalability="lab",
    source="Gao W et al. J. Nanopart. Res. 2017, 19, 37. DOI:10.1007/s11051-017-3735-7",
))

_add_sc(StructuralColorEntry(
    name="SiO2-opal-red",
    approach="bragg_opal",
    sphere_material="SiO2",
    matrix_material="air",
    diameter_nm=350.0,
    packing_fraction=0.74,
    angle_independent=False,
    published_color="red",
    published_peak_nm=630.0,
    scalability="lab",
    source="Gao W et al. J. Nanopart. Res. 2017, 19, 37. DOI:10.1007/s11051-017-3735-7",
))

_add_sc(StructuralColorEntry(
    name="PS-photonic-glass-blue",
    approach="photonic_glass",
    sphere_material="polystyrene",
    matrix_material="air",
    diameter_nm=195.0,
    packing_fraction=0.55,
    absorber="carbon_black",
    absorber_fraction=0.02,
    angle_independent=True,
    published_color="blue",
    published_peak_nm=440.0,
    scalability="lab",
    source="Park JG et al. Angew. Chem. Int. Ed. 2014, 53, 2899. DOI:10.1002/anie.201309306",
))

_add_sc(StructuralColorEntry(
    name="PS-photonic-glass-green",
    approach="photonic_glass",
    sphere_material="polystyrene",
    matrix_material="air",
    diameter_nm=240.0,
    packing_fraction=0.55,
    absorber="carbon_black",
    absorber_fraction=0.02,
    angle_independent=True,
    published_color="green",
    published_peak_nm=520.0,
    scalability="lab",
    source="Park JG et al. Angew. Chem. Int. Ed. 2014, 53, 2899. DOI:10.1002/anie.201309306",
))

_add_sc(StructuralColorEntry(
    name="SiO2-photonic-glass-blue",
    approach="photonic_glass",
    sphere_material="SiO2",
    matrix_material="water",
    diameter_nm=210.0,
    packing_fraction=0.50,
    absorber="cuttlefish_ink",
    absorber_fraction=0.03,
    angle_independent=True,
    published_color="blue",
    published_peak_nm=460.0,
    scalability="lab",
    source="Zhang Y et al. Adv. Mater. 2015, 27, 4719. DOI:10.1002/adma.201501936",
))

_add_sc(StructuralColorEntry(
    name="BCP-Cypris-green",
    approach="BCP",
    sphere_material="block_copolymer",
    matrix_material="self_assembled",
    diameter_nm=None,
    packing_fraction=None,
    angle_independent=True,
    published_color="green",
    published_peak_nm=530.0,
    scalability="pilot",
    source="Cypris Materials (now Impossible Materials). US Patent 10,730,208 B2, 2020.",
))

_add_sc(StructuralColorEntry(
    name="CNC-film-green",
    approach="CNC",
    sphere_material="cellulose_nanocrystal",
    matrix_material="self_assembled",
    diameter_nm=None,
    packing_fraction=None,
    angle_independent=False,
    published_color="iridescent green",
    published_peak_nm=550.0,
    scalability="pilot",
    source="Guidetti G et al. Adv. Mater. 2016, 28, 10042. DOI:10.1002/adma.201603386",
))


# ── Forward Model Bridge ─────────────────────────────────────────────────

def _bragg_peak(diameter_nm: float, n_sphere: float,
                n_medium: float = 1.0, ff: float = 0.74) -> Optional[float]:
    """Predict Bragg opal peak wavelength. T2: exact Bragg law."""
    try:
        from optical.bragg_opal import bragg_opal
        return bragg_opal(diameter_nm, n_sphere=n_sphere, n_medium=n_medium,
                          fill_fraction=ff)
    except (ImportError, Exception):
        # Fallback: direct calculation
        n_eff = math.sqrt(ff * n_sphere**2 + (1 - ff) * n_medium**2)
        return 1.633 * diameter_nm * n_eff


def _photonic_glass_peak(diameter_nm: float, sphere_material: str,
                          n_medium: float = 1.0, ff: float = 0.55) -> Optional[float]:
    """Predict photonic glass peak wavelength. T2: Mie + structure factor."""
    try:
        from optical.photonic_glass import photonic_glass_peak_wavelength
        return photonic_glass_peak_wavelength(diameter_nm, sphere_material, n_medium, ff)
    except (ImportError, Exception):
        # Fallback: approximate λ ≈ 2 × n_eff × d_avg
        try:
            from optical.refractive_index import n_real
            n_s = n_real(sphere_material, 550.0)
        except (ImportError, Exception):
            n_s = 1.46  # SiO2 default
        n_eff = math.sqrt(ff * n_s**2 + (1 - ff) * n_medium**2)
        d_avg = diameter_nm * (0.74 / ff) ** (1.0 / 3.0)  # rescale from FCC
        return 2.0 * n_eff * d_avg


def _spectrum_to_color(peak_nm: float) -> dict:
    """Convert a peak wavelength to approximate CIE Lab and sRGB.

    Uses a Gaussian reflectance peak centered at peak_nm with
    FWHM = 50 nm (typical for photonic glass) convolved with D65 illuminant.
    """
    try:
        import numpy as np
        from optical.cie_color import spectrum_to_XYZ, XYZ_to_Lab, XYZ_to_sRGB
        lam = np.linspace(380, 780, 81)
        fwhm = 50.0
        sigma = fwhm / 2.355
        R = 0.3 * np.exp(-0.5 * ((lam - peak_nm) / sigma) ** 2) + 0.05  # peak + background
        X, Y, Z = spectrum_to_XYZ(R, lam)
        L, a, b = XYZ_to_Lab(X, Y, Z)
        r, g, bval = XYZ_to_sRGB(X, Y, Z)
        return {"Lab": (L, a, b), "sRGB": (r, g, bval), "peak_nm": peak_nm}
    except (ImportError, Exception):
        # No CIE module → return peak only
        return {"Lab": None, "sRGB": None, "peak_nm": peak_nm}


# ── Structural Color Design Engine ────────────────────────────────────────

@dataclass
class StructuralColorSpec:
    """Specification for a structural color material design."""
    name: str = ""
    approach: str = ""
    sphere_material: str = "SiO2"
    matrix_material: str = "air"
    diameter_nm: Optional[float] = None
    packing_fraction: Optional[float] = None
    absorber: str = ""
    absorber_fraction: Optional[float] = None
    angle_independent: bool = True

    @classmethod
    def from_database(cls, name: str) -> 'StructuralColorSpec':
        if name not in STRUCTURAL_COLOR_DB:
            raise KeyError(f"Unknown structural color '{name}'. "
                           f"Known: {sorted(STRUCTURAL_COLOR_DB.keys())}")
        e = STRUCTURAL_COLOR_DB[name]
        return cls(
            name=e.name, approach=e.approach,
            sphere_material=e.sphere_material,
            matrix_material=e.matrix_material,
            diameter_nm=e.diameter_nm,
            packing_fraction=e.packing_fraction,
            absorber=e.absorber,
            absorber_fraction=e.absorber_fraction,
            angle_independent=e.angle_independent,
        )


@dataclass
class StructuralColorDesign(MaterialDesign):
    """Complete structural color material design."""
    spec: StructuralColorSpec = field(default_factory=StructuralColorSpec)
    target: TargetSpec = field(default_factory=TargetSpec)
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)

    # Optical results
    predicted_peak_nm: Optional[float] = None
    predicted_Lab: Optional[tuple] = None
    predicted_sRGB: Optional[tuple] = None
    delta_E: Optional[float] = None
    color_name: str = ""

    def material_class(self) -> str:
        return f"StructuralColor ({self.spec.approach})"

    def design_summary(self) -> str:
        lines = [f"Structural color: {self.spec.name} ({self.spec.approach})"]
        if self.predicted_peak_nm:
            lines.append(f"  Predicted peak: {self.predicted_peak_nm:.0f} nm")
        if self.predicted_sRGB:
            lines.append(f"  sRGB: {self.predicted_sRGB}")
        if self.delta_E is not None:
            lines.append(f"  ΔE from target: {self.delta_E:.1f}")
        return "\n".join(lines)

    def predict_performance(self, target: TargetSpec = None) -> PerformanceMetrics:
        if target is None:
            target = self.target
        self.target = target
        s = self.spec

        # Forward model: predict peak wavelength
        if s.approach == "bragg_opal" and s.diameter_nm is not None:
            try:
                from optical.refractive_index import n_real
                n_s = n_real(s.sphere_material, 550.0)
            except (ImportError, Exception):
                n_s = 1.46
            n_m = 1.0 if s.matrix_material == "air" else 1.33
            ff = s.packing_fraction or 0.74
            self.predicted_peak_nm = _bragg_peak(s.diameter_nm, n_s, n_m, ff)

        elif s.approach == "photonic_glass" and s.diameter_nm is not None:
            n_m = 1.0 if s.matrix_material == "air" else 1.33
            ff = s.packing_fraction or 0.55
            self.predicted_peak_nm = _photonic_glass_peak(
                s.diameter_nm, s.sphere_material, n_m, ff
            )

        elif s.approach in ("BCP", "CNC"):
            # Use published peak directly (no forward model for these)
            entry = STRUCTURAL_COLOR_DB.get(s.name)
            if entry and entry.published_peak_nm:
                self.predicted_peak_nm = entry.published_peak_nm

        # CIE color from peak
        if self.predicted_peak_nm is not None:
            color = _spectrum_to_color(self.predicted_peak_nm)
            self.predicted_Lab = color.get("Lab")
            self.predicted_sRGB = color.get("sRGB")

        # ΔE from target wavelength
        self.delta_E = None
        if target.target_wavelength_nm > 0 and self.predicted_peak_nm is not None:
            # Simple wavelength ΔE proxy: |predicted - target| mapped to ΔE
            # 1 nm wavelength shift ≈ 0.5-2.0 ΔE depending on region
            nm_diff = abs(self.predicted_peak_nm - target.target_wavelength_nm)
            self.delta_E = nm_diff * 1.0  # approximate 1:1 mapping
            # If we have full Lab for both, use proper ΔE
            if self.predicted_Lab is not None and target.target_wavelength_nm > 0:
                target_color = _spectrum_to_color(target.target_wavelength_nm)
                if target_color.get("Lab") is not None:
                    try:
                        from optical.cie_color import cie_delta_E
                        self.delta_E = cie_delta_E(
                            self.predicted_Lab, target_color["Lab"]
                        )
                    except (ImportError, Exception):
                        pass  # keep wavelength-based estimate

        # Map to PerformanceMetrics
        # "capacity" → color accuracy (inverted ΔE: lower ΔE = better)
        color_accuracy = max(0, 100.0 - (self.delta_E or 50.0)) if self.delta_E is not None else None

        # Angle independence as a quality metric
        angle_score = 1.0 if s.angle_independent else 0.3

        # Scalability
        entry = STRUCTURAL_COLOR_DB.get(s.name)
        scale_map = {"lab": 0.3, "pilot": 0.6, "industrial": 0.9}
        scalability = scale_map.get(entry.scalability if entry else "lab", 0.3)

        # Cost (T1 where published, else None)
        cost_map = {
            "bragg_opal": TieredValue(50.0, DataTier.T2_SOLID,
                "Colloidal SiO2/PS: commodity materials"),
            "photonic_glass": TieredValue(80.0, DataTier.T2_SOLID,
                "Photonic glass: colloidal + absorber"),
            "BCP": TieredValue(200.0, DataTier.T2_SOLID,
                "Block copolymer: specialty polymer"),
            "CNC": TieredValue(30.0, DataTier.T2_SOLID,
                "Cellulose nanocrystals: bio-derived, scalable"),
        }
        cost = cost_map.get(s.approach)

        # Environmental
        env_map = {
            "bragg_opal": 0.8,
            "photonic_glass": 0.7,
            "BCP": 0.4,
            "CNC": 0.95,  # bio-derived, biodegradable
        }
        env = env_map.get(s.approach, 0.5)

        self.metrics = PerformanceMetrics(
            capacity_mg_g=TieredValue(color_accuracy, DataTier.T2_SOLID,
                f"Color accuracy: 100-ΔE, peak={self.predicted_peak_nm}nm")
                if color_accuracy is not None else None,
            selectivity_ratio=TieredValue(angle_score * 100, DataTier.T1_KNOWN,
                "Angle-independent" if s.angle_independent else "Iridescent"),
            kinetics_t90_min=None,  # not applicable
            cost_per_kg_usd=cost,
            scalability_score=TieredValue(scalability, DataTier.T2_SOLID,
                f"{s.approach} process maturity"),
            durability_cycles=None,
            environmental_score=TieredValue(env, DataTier.T2_SOLID,
                f"{s.approach} environmental impact"),
        )

        return self.metrics


def design_structural_color(name: str, target: TargetSpec) -> StructuralColorDesign:
    """Design a structural color material."""
    spec = StructuralColorSpec.from_database(name)
    design = StructuralColorDesign(spec=spec)
    design.predict_performance(target)
    return design


def screen_structural_colors(target: TargetSpec,
                               top_n: int = 10) -> List[StructuralColorDesign]:
    """Screen all known structural color systems against a target."""
    designs = []
    for name in STRUCTURAL_COLOR_DB:
        try:
            d = design_structural_color(name, target)
            designs.append(d)
        except Exception:
            pass
    designs.sort(key=lambda d: d.metrics.composite_score(), reverse=True)
    return designs[:top_n]
