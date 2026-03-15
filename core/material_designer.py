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

        # 6. Equilibrium capacity at target concentration
        C_eq = 0.1  # mM default (typical trace contaminant)
        q_eq = langmuir_capacity(self.q_max_mg_g, self.K_L, C_eq)

        # 7. Selectivity (T3: size + electronic)
        selectivity = 1.0
        if self.pore_accessible and target.interferent_species:
            # Size exclusion: if interferent is larger than pore, infinite selectivity
            # Otherwise, selectivity from K ratio
            selectivity = 10.0  # default moderate selectivity
            if target.target_charge != 0:
                selectivity *= 2.0  # charge helps

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
            selectivity_ratio=TieredValue(selectivity, DataTier.T3_CONCEPTUAL,
                "Size exclusion + charge", 50),
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
                    include_polymers: bool = False,     # M2 — future
                    include_composites: bool = False,    # M3 — future
                    include_structural_color: bool = False,  # M4 — future
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

    # M2-M4: stubs for future modules
    # if include_polymers: ...
    # if include_composites: ...
    # if include_structural_color: ...

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
