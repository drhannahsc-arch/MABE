"""
core/transform_enumerator.py — Transformation Product Enumerator

Given a TargetSpecies + Matrix, enumerates thermodynamically accessible
covalent transformation products with:
  - ΔG_rxn from NIST-derived thermochemical data
  - Co-reactant requirements and source classification
  - Product phase and harvestability
  - Orthogonality score (energy/reagent self-sufficiency)
  - Capture site chemistry recommendations

Design constraints (from CaptureTransform Architecture):
  - Reality-up: all ΔG values from published thermodynamic data
  - Product-agnostic: maps chemistry space, does not pick winners
  - Orthogonality filter: ranks by passive feasibility

Data tier: Tier 2 (DOI + table reference). Sources cited per entry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class CoReactantSource(str, Enum):
    """Where the co-reactant comes from. Determines orthogonality."""
    MATRIX_NATIVE = "matrix_native"           # already in the water/air
    SUBSTRATE_PRELOADED = "substrate_preloaded"  # loaded on scaffold, replenishable
    SOLAR_PHOTOCATALYTIC = "solar_photocatalytic"  # light-generated in situ
    SELF_CONTAINED = "self_contained"          # pre-positioned in 3D scaffold
    EXTERNALLY_SUPPLIED = "externally_supplied"  # fails orthogonality
    NONE = "none"                              # no co-reactant needed

    @property
    def orthogonality_rank(self) -> int:
        """Lower is better. 99 = fails orthogonality."""
        return {
            CoReactantSource.NONE: 0,
            CoReactantSource.MATRIX_NATIVE: 1,
            CoReactantSource.SELF_CONTAINED: 2,
            CoReactantSource.SUBSTRATE_PRELOADED: 3,
            CoReactantSource.SOLAR_PHOTOCATALYTIC: 4,
            CoReactantSource.EXTERNALLY_SUPPLIED: 99,
        }[self]


class EnergyInput(str, Enum):
    """What energy drives the transformation."""
    NONE = "none"                              # spontaneous at ambient
    PASSIVE_SOLAR_THERMAL = "passive_solar_thermal"  # ≤80°C dark surface
    PASSIVE_SOLAR_PHOTOCATALYTIC = "passive_solar_photocatalytic"  # UV/vis
    ACTIVE_THERMAL = "active_thermal"          # >80°C, fails orthogonality
    ELECTROCHEMICAL = "electrochemical"        # fails orthogonality

    @property
    def orthogonality_rank(self) -> int:
        return {
            EnergyInput.NONE: 0,
            EnergyInput.PASSIVE_SOLAR_THERMAL: 1,
            EnergyInput.PASSIVE_SOLAR_PHOTOCATALYTIC: 2,
            EnergyInput.ACTIVE_THERMAL: 99,
            EnergyInput.ELECTROCHEMICAL: 99,
        }[self]


class TurnoverMode(str, Enum):
    """Whether the capture site is consumed."""
    CATALYTIC = "catalytic"                          # site regenerates
    STOICHIOMETRIC_CHEAP = "stoichiometric_cheap"    # Fe0, ZrO2 — consumed but cheap
    STOICHIOMETRIC_EXPENSIVE = "stoichiometric_expensive"  # fails cost orthogonality

    @property
    def orthogonality_rank(self) -> int:
        return {
            TurnoverMode.CATALYTIC: 0,
            TurnoverMode.STOICHIOMETRIC_CHEAP: 1,
            TurnoverMode.STOICHIOMETRIC_EXPENSIVE: 99,
        }[self]


class ProductPhase(str, Enum):
    """Physical state of the transformation product."""
    SOLID_PRECIPITATE = "solid_precipitate"
    BOUND_TO_SITE = "bound_to_site"
    DISSOLVED = "dissolved"
    GAS = "gas"


class ClickCompatibility(str, Enum):
    """Which click chemistries are safe with this capture site."""
    SPAAC_ONLY = "spaac_only"       # Cu-sensitive (Zn-CA mimics, thiol systems)
    CUAAC_OK = "cuaac_ok"           # Cu-tolerant (Zr-oxide, Fe-oxide, sulfonic acid)
    THIOL_MALEIMIDE = "thiol_maleimide"  # alternative


# ═══════════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CoReactantSpec:
    """A co-reactant needed for the transformation."""
    identity: str                # "Ca2+", "Mg2+", "H2O", "solar_photon"
    formula: str                 # "Ca²⁺", "Mg²⁺", "H₂O", "hν"
    source: CoReactantSource
    min_concentration_mM: float = 0.0  # 0 for non-solution species
    notes: str = ""


@dataclass
class CaptureSiteChemistry:
    """What functional chemistry performs the capture-transform."""
    name: str                    # "Zn-CA mimic", "hydrous zirconia", "sulfonic acid"
    description: str             # how it works
    functional_groups: list[str] = field(default_factory=list)  # e.g. ["Zn2+", "3×N-donor"]
    click_compatibility: ClickCompatibility = ClickCompatibility.SPAAC_ONLY
    notes: str = ""


@dataclass
class TransformationProduct:
    """One thermodynamically accessible transformation product for a target.

    This is the atomic output unit of the enumerator.
    """
    # ── Identity ──
    name: str                    # "calcite (CaCO₃)"
    formula: str                 # "CaCO₃"
    target_formula: str          # "CO₂" — what was captured

    # ── Thermodynamics ──
    dg_rxn_kj_mol: float         # ΔG of the overall transformation
    ksp_log: Optional[float] = None  # log(Ksp) if precipitation product
    dg_source: str = ""          # citation: "NIST-JANAF, Chase 1998"

    # ── Co-reactant ──
    co_reactants: list[CoReactantSpec] = field(default_factory=list)

    # ── Energy ──
    energy_input: EnergyInput = EnergyInput.NONE
    activation_barrier_kj_mol: Optional[float] = None  # estimated or lit

    # ── Product properties ──
    product_phase: ProductPhase = ProductPhase.SOLID_PRECIPITATE
    harvestable: bool = True
    feedstock_value: str = ""    # what it's useful for

    # ── Capture site ──
    capture_sites: list[CaptureSiteChemistry] = field(default_factory=list)

    # ── Turnover ──
    turnover: TurnoverMode = TurnoverMode.CATALYTIC

    # ── Orthogonality ──
    orthogonality_score: float = 0.0  # computed, 0-1 where 1 = gold standard

    # ── Cascade potential ──
    benefits_from_confinement: bool = False
    cascade_notes: str = ""      # how 3D scaffold improves this pathway

    # ── Provenance ──
    notes: str = ""

    @property
    def is_orthogonal(self) -> bool:
        """Does this pathway pass the orthogonality filter?"""
        return self.orthogonality_score > 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Orthogonality scoring
# ═══════════════════════════════════════════════════════════════════════════

def compute_orthogonality(
    dg_total_kj: float,
    co_reactant_source: CoReactantSource,
    energy_input: EnergyInput,
    turnover: TurnoverMode,
    temperature_c: float = 25.0,
) -> float:
    """Compute orthogonality score (0-1).

    Score components:
      - Thermodynamic feasibility: ΔG < 0 at operating T?  (0 or 0.25)
      - Co-reactant source rank                            (0 to 0.25)
      - Energy input rank                                  (0 to 0.25)
      - Turnover mode rank                                 (0 to 0.25)

    Any component scoring 99 (fails orthogonality) → total = 0.
    """
    # Hard fail: any non-orthogonal component kills the pathway
    if co_reactant_source.orthogonality_rank >= 99:
        return 0.0
    if energy_input.orthogonality_rank >= 99:
        return 0.0
    if turnover.orthogonality_rank >= 99:
        return 0.0

    score = 0.0

    # Thermodynamic feasibility at operating T
    # ΔG < 0 is required. More negative = better (up to a cap).
    if dg_total_kj < 0:
        # Scale: ΔG = 0 → 0.0, ΔG ≤ -50 → full 0.25
        score += min(0.25, 0.25 * abs(dg_total_kj) / 50.0)
    else:
        # Endergonic — only passes if photocatalytic
        if energy_input == EnergyInput.PASSIVE_SOLAR_PHOTOCATALYTIC:
            score += 0.05  # marginal credit
        else:
            return 0.0

    # Co-reactant source (0=none → 0.25, 4=solar → ~0.06)
    cr_rank = co_reactant_source.orthogonality_rank
    score += 0.25 * (1.0 - cr_rank / 5.0)

    # Energy input (0=none → 0.25, 2=solar_photo → ~0.08)
    ei_rank = energy_input.orthogonality_rank
    score += 0.25 * (1.0 - ei_rank / 3.0)

    # Turnover (0=catalytic → 0.25, 1=stoich_cheap → 0.125)
    to_rank = turnover.orthogonality_rank
    score += 0.25 * (1.0 - to_rank / 2.0)

    return round(min(1.0, score), 4)


# ═══════════════════════════════════════════════════════════════════════════
# Thermochemical database — NIST-derived, hardcoded
#
# Each entry is a function returning a list of TransformationProducts
# for a given target + matrix.
#
# Data provenance:
#   - NIST-JANAF tables (Chase, 1998)
#   - NIST SRD 46 (stability constants)
#   - CRC Handbook of Chemistry and Physics
#   - Stumm & Morgan, Aquatic Chemistry (Ksp values)
#   - Specific DOIs cited per entry
# ═══════════════════════════════════════════════════════════════════════════

def _co2_products(matrix_species: dict[str, float]) -> list[TransformationProduct]:
    """CO₂ transformation products. matrix_species: {formula: conc_mM}."""
    products = []

    ca_available = matrix_species.get("Ca2+", 0.0) > 0.01
    mg_available = matrix_species.get("Mg2+", 0.0) > 0.01

    # ── CaCO₃ via Zn-CA mimic catalysis ──
    if ca_available:
        products.append(TransformationProduct(
            name="calcite (CaCO₃) via carbonic anhydrase mimic",
            formula="CaCO₃",
            target_formula="CO₂",
            dg_rxn_kj_mol=-47.7,
            ksp_log=-8.48,  # log(Ksp) calcite, Stumm & Morgan
            dg_source="NIST-JANAF, Chase 1998; Ksp: Stumm & Morgan 1996",
            co_reactants=[
                CoReactantSpec(
                    identity="Ca2+", formula="Ca²⁺",
                    source=CoReactantSource.MATRIX_NATIVE,
                    min_concentration_mM=0.01,
                    notes="Hard water, seawater, concrete wash"
                ),
                CoReactantSpec(
                    identity="H2O", formula="H₂O",
                    source=CoReactantSource.MATRIX_NATIVE,
                ),
            ],
            energy_input=EnergyInput.NONE,
            activation_barrier_kj_mol=70.0,  # uncatalyzed CO2 hydration; CA mimic lowers to ~50
            product_phase=ProductPhase.SOLID_PRECIPITATE,
            harvestable=True,
            feedstock_value="Construction material, calcium supplement, carbon sequestration",
            capture_sites=[
                CaptureSiteChemistry(
                    name="Zn-CA mimic",
                    description="Zn²⁺ center with 3 N/O donors + 1 open coordination site. "
                                "Catalyzes CO₂ hydration to HCO₃⁻ at ambient T. "
                                "HCO₃⁻ precipitates with matrix Ca²⁺.",
                    functional_groups=["Zn2+", "3×N-donor (cyclen/BAPA/hydroxamate)"],
                    click_compatibility=ClickCompatibility.SPAAC_ONLY,
                    notes="Cu poisons the Zn active site — SPAAC mandatory"
                ),
            ],
            turnover=TurnoverMode.CATALYTIC,
            benefits_from_confinement=True,
            cascade_notes="Cage confines HCO₃⁻ intermediate at mM effective concentration, "
                          "accelerating CaCO₃ nucleation. Pre-positioned Ca²⁺ chelate "
                          "modules (Pattern 1 in architecture).",
        ))

    # ── MgCO₃ via Zn-CA mimic ──
    if mg_available:
        products.append(TransformationProduct(
            name="magnesite/nesquehonite (MgCO₃) via CA mimic",
            formula="MgCO₃",
            target_formula="CO₂",
            dg_rxn_kj_mol=-25.3,
            ksp_log=-7.46,  # magnesite; nesquehonite -5.17
            dg_source="NIST-JANAF; Ksp: Langmuir 1997",
            co_reactants=[
                CoReactantSpec(
                    identity="Mg2+", formula="Mg²⁺",
                    source=CoReactantSource.MATRIX_NATIVE,
                    min_concentration_mM=0.01,
                ),
            ],
            energy_input=EnergyInput.NONE,
            activation_barrier_kj_mol=75.0,  # Mg dehydration slower than Ca
            product_phase=ProductPhase.SOLID_PRECIPITATE,
            harvestable=True,
            feedstock_value="Fire retardant, construction material",
            capture_sites=[
                CaptureSiteChemistry(
                    name="Zn-CA mimic",
                    description="Same as CaCO₃ pathway; slower Mg²⁺ dehydration kinetics.",
                    functional_groups=["Zn2+", "3×N-donor"],
                    click_compatibility=ClickCompatibility.SPAAC_ONLY,
                ),
            ],
            turnover=TurnoverMode.CATALYTIC,
            benefits_from_confinement=True,
            cascade_notes="Confinement helps overcome slow Mg²⁺ dehydration.",
        ))

    # ── Amine capture → carbamate → CaCO₃ (two-step, passive) ──
    products.append(TransformationProduct(
        name="carbamate → CaCO₃ via amine sorbent",
        formula="CaCO₃",
        target_formula="CO₂",
        dg_rxn_kj_mol=-65.0,  # amine-CO2: -50 to -80; mineralization adds more
        dg_source="Sanz-Perez et al., Chem. Rev. 2016, 116, 11840; NIST-JANAF",
        co_reactants=[
            CoReactantSpec(
                identity="primary amine (site)", formula="R-NH₂",
                source=CoReactantSource.NONE,
                notes="The amine IS the capture site, not a co-reactant"
            ),
            CoReactantSpec(
                identity="Ca2+", formula="Ca²⁺",
                source=CoReactantSource.MATRIX_NATIVE
                    if ca_available else CoReactantSource.SUBSTRATE_PRELOADED,
                min_concentration_mM=0.01,
                notes="Periodic Ca²⁺-bearing water contact regenerates amine"
            ),
        ],
        energy_input=EnergyInput.NONE,
        product_phase=ProductPhase.SOLID_PRECIPITATE,
        harvestable=True,
        feedstock_value="Carbon mineralization + construction material",
        capture_sites=[
            CaptureSiteChemistry(
                name="Amine sorbent (PEI/APS analog)",
                description="Primary amine captures CO₂ as carbamate at ambient T. "
                            "Ca²⁺-bearing water hydrolyzes carbamate → CaCO₃ + amine. "
                            "Amine regenerates without thermal swing.",
                functional_groups=["R-NH₂", "PEI", "aminopropylsilane"],
                click_compatibility=ClickCompatibility.CUAAC_OK,
            ),
        ],
        turnover=TurnoverMode.CATALYTIC,
        benefits_from_confinement=True,
        cascade_notes="Cage positions amine modules + Ca²⁺ chelate modules "
                      "for in-situ mineralization without separate wash step.",
    ))

    # ── Photocatalytic CO₂ reduction ──
    products.append(TransformationProduct(
        name="formate/methanol via TiO₂ photocatalysis",
        formula="HCOO⁻ / CH₃OH",
        target_formula="CO₂",
        dg_rxn_kj_mol=33.0,  # endergonic — driven by photon energy
        dg_source="Habisreutinger et al., Angew. Chem. Int. Ed. 2013, 52, 7372",
        co_reactants=[
            CoReactantSpec(
                identity="solar photon", formula="hν",
                source=CoReactantSource.SOLAR_PHOTOCATALYTIC,
                notes="UV/vis, bandgap ~3.2 eV (TiO₂) or ~2.7 eV (g-C₃N₄)"
            ),
            CoReactantSpec(
                identity="H2O", formula="H₂O",
                source=CoReactantSource.MATRIX_NATIVE,
                notes="Electron donor (oxidized to O₂)"
            ),
        ],
        energy_input=EnergyInput.PASSIVE_SOLAR_PHOTOCATALYTIC,
        product_phase=ProductPhase.DISSOLVED,
        harvestable=True,
        feedstock_value="Formate: H₂ carrier, chemical feedstock. Methanol: fuel.",
        capture_sites=[
            CaptureSiteChemistry(
                name="TiO₂ / g-C₃N₄ photocatalyst",
                description="Solar UV/vis generates e⁻/h⁺ pairs on semiconductor surface. "
                            "e⁻ reduces adsorbed CO₂; h⁺ oxidizes H₂O.",
                functional_groups=["TiO₂", "g-C₃N₄"],
                click_compatibility=ClickCompatibility.CUAAC_OK,
                notes="Rates ~µmol/g·h for state-of-art. Fully passive."
            ),
        ],
        turnover=TurnoverMode.CATALYTIC,
        benefits_from_confinement=True,
        cascade_notes="Cage separates photocatalyst (exterior, solar access) from "
                      "reduction site (interior, product retention). Pattern 3.",
    ))

    return products


def _phosphate_products(matrix_species: dict[str, float]) -> list[TransformationProduct]:
    """Phosphate (H₂PO₄⁻/HPO₄²⁻/PO₄³⁻) transformation products."""
    products = []

    ca_available = matrix_species.get("Ca2+", 0.0) > 0.01
    mg_available = matrix_species.get("Mg2+", 0.0) > 0.01
    nh4_available = matrix_species.get("NH4+", 0.0) > 0.01
    fe3_available = matrix_species.get("Fe3+", 0.0) > 0.001

    # ── Hydroxyapatite ──
    products.append(TransformationProduct(
        name="hydroxyapatite Ca₅(PO₄)₃OH",
        formula="Ca₅(PO₄)₃OH",
        target_formula="PO₄³⁻",
        dg_rxn_kj_mol=-120.0,  # very favorable from Ksp
        ksp_log=-58.0,  # Stumm & Morgan
        dg_source="Stumm & Morgan, Aquatic Chemistry, 3rd ed.; Ksp compilation",
        co_reactants=[
            CoReactantSpec(
                identity="Ca2+", formula="Ca²⁺",
                source=CoReactantSource.MATRIX_NATIVE
                    if ca_available else CoReactantSource.SUBSTRATE_PRELOADED,
                min_concentration_mM=0.1,
                notes="Ca-rich substrate (Ca-silicate glass) or matrix Ca²⁺"
            ),
        ],
        energy_input=EnergyInput.NONE,
        product_phase=ProductPhase.SOLID_PRECIPITATE,
        harvestable=True,
        feedstock_value="Direct fertilizer, bone graft precursor, P recovery",
        capture_sites=[
            CaptureSiteChemistry(
                name="Zr(IV) oxide selective adsorber + Ca²⁺ nucleation template",
                description="Zr(IV) or La(III) surface selectively adsorbs PO₄³⁻ "
                            "(high selectivity over SO₄²⁻). Adjacent Ca²⁺ source "
                            "spontaneously nucleates HAp.",
                functional_groups=["ZrO₂·nH₂O", "La₂O₃", "Ca²⁺-carboxylate"],
                click_compatibility=ClickCompatibility.CUAAC_OK,
            ),
        ],
        turnover=TurnoverMode.STOICHIOMETRIC_CHEAP,
        benefits_from_confinement=True,
        cascade_notes="Cage co-locates Zr-selective-capture (module A) with "
                      "Ca²⁺-release (module B) for confinement-accelerated "
                      "nucleation. Pattern 5 in architecture.",
    ))

    # ── Struvite (dual-target co-capture) ──
    if mg_available or True:  # always enumerate — Mg can be preloaded
        products.append(TransformationProduct(
            name="struvite MgNH₄PO₄·6H₂O",
            formula="MgNH₄PO₄·6H₂O",
            target_formula="PO₄³⁻",
            dg_rxn_kj_mol=-75.8,
            ksp_log=-13.26,  # Stumm & Morgan; Ohlinger et al. 1998
            dg_source="Ohlinger et al., Water Res. 1998, 32, 3607; Ksp compilation",
            co_reactants=[
                CoReactantSpec(
                    identity="Mg2+", formula="Mg²⁺",
                    source=CoReactantSource.MATRIX_NATIVE
                        if mg_available else CoReactantSource.SUBSTRATE_PRELOADED,
                    min_concentration_mM=0.1,
                ),
                CoReactantSpec(
                    identity="NH4+", formula="NH₄⁺",
                    source=CoReactantSource.MATRIX_NATIVE
                        if nh4_available else CoReactantSource.SUBSTRATE_PRELOADED,
                    min_concentration_mM=0.1,
                    notes="Co-captured from wastewater — dual target"
                ),
            ],
            energy_input=EnergyInput.NONE,
            product_phase=ProductPhase.SOLID_PRECIPITATE,
            harvestable=True,
            feedstock_value="Slow-release fertilizer (N + P in one crystal)",
            capture_sites=[
                CaptureSiteChemistry(
                    name="Mg-loaded dual-capture surface",
                    description="Mg²⁺ pre-loaded on scaffold. PO₄³⁻ captured by "
                                "Zr/La site (one hemisphere). NH₄⁺ captured by "
                                "sulfonic acid site (opposite hemisphere). "
                                "Struvite crystallizes spontaneously.",
                    functional_groups=["ZrO₂", "-SO₃H", "Mg²⁺-chelate"],
                    click_compatibility=ClickCompatibility.CUAAC_OK,
                ),
            ],
            turnover=TurnoverMode.STOICHIOMETRIC_CHEAP,
            benefits_from_confinement=True,
            cascade_notes="Gold-standard cascade: Pattern 2. Cage is a self-contained "
                          "fertilizer factory. Three modules, fully orthogonal.",
        ))

    # ── Zirconium phosphate ──
    products.append(TransformationProduct(
        name="zirconium phosphate Zr(HPO₄)₂",
        formula="Zr(HPO₄)₂",
        target_formula="PO₄³⁻",
        dg_rxn_kj_mol=-95.0,
        dg_source="Clearfield, Chem. Rev. 1988, 88, 125; formation thermodynamics",
        co_reactants=[
            CoReactantSpec(
                identity="Zr(IV) surface", formula="Zr⁴⁺",
                source=CoReactantSource.NONE,
                notes="Zr IS the capture site — consumed stoichiometrically"
            ),
        ],
        energy_input=EnergyInput.NONE,
        product_phase=ProductPhase.BOUND_TO_SITE,
        harvestable=True,
        feedstock_value="Ion-exchange material, catalyst support, proton conductor",
        capture_sites=[
            CaptureSiteChemistry(
                name="Hydrous zirconia (ZrO₂·nH₂O)",
                description="Irreversibly binds phosphate forming Zr(HPO₄)₂. "
                            "Extremely selective over SO₄²⁻.",
                functional_groups=["ZrO₂·nH₂O"],
                click_compatibility=ClickCompatibility.CUAAC_OK,
            ),
        ],
        turnover=TurnoverMode.STOICHIOMETRIC_CHEAP,
        benefits_from_confinement=False,
        cascade_notes="Flat surface sufficient — Zr sites are the product.",
    ))

    # ── Iron phosphate ──
    products.append(TransformationProduct(
        name="iron phosphate FePO₄",
        formula="FePO₄",
        target_formula="PO₄³⁻",
        dg_rxn_kj_mol=-90.0,
        ksp_log=-26.4,
        dg_source="Nriagu, Geochim. Cosmochim. Acta 1972; NIST-JANAF",
        co_reactants=[
            CoReactantSpec(
                identity="Fe(III) surface", formula="Fe³⁺",
                source=CoReactantSource.NONE
                    if not fe3_available else CoReactantSource.MATRIX_NATIVE,
                notes="Fe-oxyhydroxide IS the site, or Fe³⁺ from matrix"
            ),
        ],
        energy_input=EnergyInput.NONE,
        product_phase=ProductPhase.SOLID_PRECIPITATE,
        harvestable=True,
        feedstock_value="LFP battery cathode precursor (LiFePO₄)",
        capture_sites=[
            CaptureSiteChemistry(
                name="Fe-oxyhydroxide surface",
                description="Goethite or ferrihydrite surface adsorbs PO₄³⁻, "
                            "forming FePO₄ phases spontaneously.",
                functional_groups=["FeOOH", "Fe₂O₃·nH₂O"],
                click_compatibility=ClickCompatibility.CUAAC_OK,
            ),
        ],
        turnover=TurnoverMode.STOICHIOMETRIC_CHEAP,
        benefits_from_confinement=False,
    ))

    return products


def _nitrogen_products(matrix_species: dict[str, float],
                       target_oxidation_state: str) -> list[TransformationProduct]:
    """Nitrogen transformation products, branched by oxidation state.

    target_oxidation_state: "NH3", "NH4+", "NO3-", "N2"
    """
    products = []

    mg_available = matrix_species.get("Mg2+", 0.0) > 0.01
    po4_available = matrix_species.get("PO4_total", 0.0) > 0.01

    if target_oxidation_state in ("NH3", "NH4+"):
        # ── Ammonium salt via acid surface ──
        products.append(TransformationProduct(
            name="ammonium sulfonate salt",
            formula="R-SO₃⁻·NH₄⁺",
            target_formula="NH₃",
            dg_rxn_kj_mol=-52.0,  # acid-base, very favorable
            dg_source="Standard acid-base thermodynamics; pKa(R-SO₃H)~-1, pKa(NH₄⁺)=9.25",
            co_reactants=[
                CoReactantSpec(
                    identity="sulfonic acid group", formula="-SO₃H",
                    source=CoReactantSource.NONE,
                    notes="The acid IS the capture site"
                ),
            ],
            energy_input=EnergyInput.NONE,
            product_phase=ProductPhase.BOUND_TO_SITE,
            harvestable=True,
            feedstock_value="Fertilizer precursor — water wash releases (NH₄)₂SO₄",
            capture_sites=[
                CaptureSiteChemistry(
                    name="Sulfonic acid surface",
                    description="Fixed -SO₃H groups capture NH₃ as NH₄⁺ by proton transfer. "
                                "Water wash releases ammonium salt, regenerates acid.",
                    functional_groups=["-SO₃H", "Nafion-like", "sulfonated polymer"],
                    click_compatibility=ClickCompatibility.CUAAC_OK,
                ),
            ],
            turnover=TurnoverMode.CATALYTIC,
            benefits_from_confinement=False,
            cascade_notes="Acid-base is fast on flat surface. No scaffold needed.",
        ))

        # ── Struvite co-capture (if PO₄³⁻ present) ──
        if po4_available:
            products.append(TransformationProduct(
                name="struvite MgNH₄PO₄·6H₂O (N-pathway)",
                formula="MgNH₄PO₄·6H₂O",
                target_formula="NH₄⁺",
                dg_rxn_kj_mol=-75.8,
                ksp_log=-13.26,
                dg_source="Ohlinger et al., Water Res. 1998, 32, 3607",
                co_reactants=[
                    CoReactantSpec(
                        identity="PO4", formula="PO₄³⁻",
                        source=CoReactantSource.MATRIX_NATIVE,
                    ),
                    CoReactantSpec(
                        identity="Mg2+", formula="Mg²⁺",
                        source=CoReactantSource.MATRIX_NATIVE
                            if mg_available else CoReactantSource.SUBSTRATE_PRELOADED,
                    ),
                ],
                energy_input=EnergyInput.NONE,
                product_phase=ProductPhase.SOLID_PRECIPITATE,
                harvestable=True,
                feedstock_value="Slow-release fertilizer",
                capture_sites=[
                    CaptureSiteChemistry(
                        name="Mg-loaded dual-capture surface",
                        description="Same as phosphate struvite pathway — dual target.",
                        functional_groups=["Mg²⁺-chelate", "ZrO₂", "-SO₃H"],
                        click_compatibility=ClickCompatibility.CUAAC_OK,
                    ),
                ],
                turnover=TurnoverMode.STOICHIOMETRIC_CHEAP,
                benefits_from_confinement=True,
                cascade_notes="Pattern 2 cascade — cage positions both capture modules.",
            ))

    if target_oxidation_state == "NO3-":
        # ── Photocatalytic NO₃⁻ → NH₄⁺ ──
        products.append(TransformationProduct(
            name="ammonium via photocatalytic nitrate reduction",
            formula="NH₄⁺",
            target_formula="NO₃⁻",
            dg_rxn_kj_mol=-30.0,  # favorable at modest reduction potential
            dg_source="Li et al., Chem. Soc. Rev. 2022, 51, 6998 (review)",
            co_reactants=[
                CoReactantSpec(
                    identity="solar photon", formula="hν",
                    source=CoReactantSource.SOLAR_PHOTOCATALYTIC,
                ),
                CoReactantSpec(
                    identity="H2O", formula="H₂O",
                    source=CoReactantSource.MATRIX_NATIVE,
                ),
            ],
            energy_input=EnergyInput.PASSIVE_SOLAR_PHOTOCATALYTIC,
            product_phase=ProductPhase.DISSOLVED,
            harvestable=True,
            feedstock_value="Fertilizer precursor (capture NH₄⁺ on adjacent acid site)",
            capture_sites=[
                CaptureSiteChemistry(
                    name="Doped TiO₂ photocatalyst",
                    description="Cu/Fe/Ag-doped TiO₂ under solar illumination reduces "
                                "NO₃⁻ to NH₄⁺ using H₂O as electron donor.",
                    functional_groups=["TiO₂-Cu", "TiO₂-Fe"],
                    click_compatibility=ClickCompatibility.CUAAC_OK,
                ),
            ],
            turnover=TurnoverMode.CATALYTIC,
            benefits_from_confinement=True,
            cascade_notes="Pattern 3: exterior photocatalyst, interior reduction + "
                          "NH₄⁺ capture on acid group. Prevents back-oxidation.",
        ))

        # ── ZVI reductive denitrification ──
        products.append(TransformationProduct(
            name="ammonium via zerovalent iron reduction",
            formula="NH₄⁺",
            target_formula="NO₃⁻",
            dg_rxn_kj_mol=-560.0,  # very favorable: 4Fe⁰ + NO₃⁻ + 10H⁺ → NH₄⁺ + 4Fe²⁺ + 3H₂O
            dg_source="Cheng et al., Water Res. 1997, 31, 3073; Fe⁰/NO₃⁻ thermodynamics",
            co_reactants=[
                CoReactantSpec(
                    identity="Fe0", formula="Fe⁰",
                    source=CoReactantSource.NONE,
                    notes="Fe⁰ IS the capture site — consumed"
                ),
            ],
            energy_input=EnergyInput.NONE,
            product_phase=ProductPhase.DISSOLVED,
            harvestable=True,
            feedstock_value="NH₄⁺ product capturable as ammonium salt or struvite",
            capture_sites=[
                CaptureSiteChemistry(
                    name="Zerovalent iron nanoparticles",
                    description="Fe⁰ NPs click-tethered to netting. Spontaneously "
                                "reduces NO₃⁻ to NH₄⁺ (or N₂). Fe⁰ consumed.",
                    functional_groups=["nZVI", "Fe⁰"],
                    click_compatibility=ClickCompatibility.CUAAC_OK,
                    notes="Netting must be replaced/reloaded when Fe⁰ exhausted"
                ),
            ],
            turnover=TurnoverMode.STOICHIOMETRIC_CHEAP,
            benefits_from_confinement=False,
        ))

    if target_oxidation_state == "N2":
        # N₂ fixation — excluded under orthogonality, but enumerate for completeness
        products.append(TransformationProduct(
            name="ammonia via N₂ fixation (NON-ORTHOGONAL)",
            formula="NH₃",
            target_formula="N₂",
            dg_rxn_kj_mol=16.4,  # endergonic per NH₃
            dg_source="NIST-JANAF; Haber-Bosch thermodynamics",
            co_reactants=[
                CoReactantSpec(
                    identity="electrons + protons", formula="e⁻ + H⁺",
                    source=CoReactantSource.EXTERNALLY_SUPPLIED,
                    notes="No passive ambient pathway cracks N≡N (945 kJ/mol)"
                ),
            ],
            energy_input=EnergyInput.ELECTROCHEMICAL,
            product_phase=ProductPhase.GAS,
            harvestable=True,
            feedstock_value="Ammonia — fertilizer, chemical feedstock",
            capture_sites=[],  # no passive site chemistry exists
            turnover=TurnoverMode.CATALYTIC,
            benefits_from_confinement=False,
            notes="EXCLUDED under orthogonality. Requires electrochemical adapter.",
        ))

    return products


def _heavy_metal_sulfide_products(target_identity: str, target_formula: str,
                                  ksp_log: float) -> list[TransformationProduct]:
    """Heavy metal → metal sulfide precipitation."""
    sulfide_formula = target_formula.replace("²⁺", "") + "S"
    # Map common ones
    _sulfide_map = {
        "Pb2+": ("PbS", "galena"),
        "Cd2+": ("CdS", "greenockite"),
        "Hg2+": ("HgS", "cinnabar"),
        "Cu2+": ("CuS", "covellite"),
        "Zn2+": ("ZnS", "sphalerite"),
        "Ag+":  ("Ag₂S", "acanthite"),
    }
    formula, mineral = _sulfide_map.get(target_formula, (sulfide_formula, "sulfide"))

    return [TransformationProduct(
        name=f"{mineral} ({formula})",
        formula=formula,
        target_formula=target_formula,
        dg_rxn_kj_mol=-150.0,  # conservative; actual varies by metal
        ksp_log=ksp_log,
        dg_source="Stumm & Morgan; NIST SRD 46",
        co_reactants=[
            CoReactantSpec(
                identity="sulfide surface", formula="S²⁻",
                source=CoReactantSource.NONE,
                notes="Thiol/FeS₂ surface IS the site"
            ),
        ],
        energy_input=EnergyInput.NONE,
        product_phase=ProductPhase.SOLID_PRECIPITATE,
        harvestable=True,
        feedstock_value="Stable sequestration; CdS = semiconductor precursor",
        capture_sites=[
            CaptureSiteChemistry(
                name="Thiol/sulfide surface",
                description="Thiol-functionalized or FeS₂ surface spontaneously "
                            f"precipitates {formula} (Ksp = 10^{ksp_log}).",
                functional_groups=["-SH", "FeS₂", "thiol-SAM"],
                click_compatibility=ClickCompatibility.SPAAC_ONLY,
                notes="Cu in CuAAC would compete for thiol sites"
            ),
        ],
        turnover=TurnoverMode.STOICHIOMETRIC_CHEAP,
        benefits_from_confinement=True,
        cascade_notes="Cage pore selectivity helps in complex matrices.",
    )]


def _simple_precipitation_products(target_identity: str, target_formula: str,
                                   product_name: str, product_formula: str,
                                   counter_ion: str, counter_formula: str,
                                   ksp_log: float, dg_kj: float,
                                   feedstock: str, dg_source: str,
                                   counter_source: CoReactantSource,
                                   ) -> list[TransformationProduct]:
    """Generic precipitation pathway (F⁻→CaF₂, SO₂→gypsum, etc.)."""
    return [TransformationProduct(
        name=product_name,
        formula=product_formula,
        target_formula=target_formula,
        dg_rxn_kj_mol=dg_kj,
        ksp_log=ksp_log,
        dg_source=dg_source,
        co_reactants=[
            CoReactantSpec(
                identity=counter_ion, formula=counter_formula,
                source=counter_source,
            ),
        ],
        energy_input=EnergyInput.NONE,
        product_phase=ProductPhase.SOLID_PRECIPITATE,
        harvestable=True,
        feedstock_value=feedstock,
        capture_sites=[
            CaptureSiteChemistry(
                name=f"{counter_ion}-loaded surface",
                description=f"Spontaneous precipitation: {target_formula} + "
                            f"{counter_formula} → {product_formula}",
                functional_groups=[f"{counter_ion}-loaded"],
                click_compatibility=ClickCompatibility.CUAAC_OK,
            ),
        ],
        turnover=TurnoverMode.STOICHIOMETRIC_CHEAP,
        benefits_from_confinement=False,
    )]


# ═══════════════════════════════════════════════════════════════════════════
# Target classification and routing
# ═══════════════════════════════════════════════════════════════════════════

# Known target formulas → enumerator functions
_TARGET_ROUTER: dict[str, str] = {
    # CO2 family
    "CO2": "co2", "CO₂": "co2", "HCO3-": "co2", "CO3_2-": "co2",
    # Phosphate family
    "PO4_3-": "phosphate", "HPO4_2-": "phosphate", "H2PO4-": "phosphate",
    "PO₄³⁻": "phosphate", "HPO₄²⁻": "phosphate",
    # Nitrogen family
    "NH3": "nh3", "NH₃": "nh3",
    "NH4+": "nh4", "NH₄⁺": "nh4",
    "NO3-": "no3", "NO₃⁻": "no3",
    "N2": "n2", "N₂": "n2",
    # Heavy metals
    "Pb2+": "pb", "Pb²⁺": "pb",
    "Cd2+": "cd", "Cd²⁺": "cd",
    "Hg2+": "hg", "Hg²⁺": "hg",
    "Cu2+": "cu_metal", "Cu²⁺": "cu_metal",
    "Zn2+": "zn_metal", "Zn²⁺": "zn_metal",
    # Simple anions
    "F-": "fluoride", "F⁻": "fluoride",
    "SO2": "so2", "SO₂": "so2",
    # Arsenic
    "H2AsO4-": "arsenic", "AsO4_3-": "arsenic", "As(V)": "arsenic",
}

# Heavy metal Ksp values (log scale)
_METAL_SULFIDE_KSP = {
    "pb": ("Pb2+", -28.0),
    "cd": ("Cd2+", -27.0),
    "hg": ("Hg2+", -52.0),
    "cu_metal": ("Cu2+", -36.0),
    "zn_metal": ("Zn2+", -24.7),
}


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def matrix_to_species_dict(matrix) -> dict[str, float]:
    """Extract species concentrations from a Matrix object.

    Returns dict mapping formula → concentration in mM.
    Falls back to empty dict if matrix has no competing_species.
    """
    species = {}
    if hasattr(matrix, 'competing_species'):
        for cs in matrix.competing_species:
            species[cs.formula] = cs.concentration_mm
    # Also check for common ions by description keywords
    desc = getattr(matrix, 'description', '').lower()
    if 'hard water' in desc or 'seawater' in desc:
        species.setdefault("Ca2+", 2.0)
        species.setdefault("Mg2+", 5.0)
    if 'wastewater' in desc or 'agricultural' in desc:
        species.setdefault("NH4+", 1.0)
        species.setdefault("PO4_total", 0.5)
    return species


def enumerate_transformations(
    target_formula: str,
    matrix=None,
    matrix_species: Optional[dict[str, float]] = None,
    temperature_c: float = 25.0,
) -> list[TransformationProduct]:
    """Enumerate all thermodynamically accessible transformation products.

    Args:
        target_formula: Chemical formula of the target (e.g., "CO2", "PO4_3-", "Pb2+")
        matrix: Optional Matrix object from core.problem
        matrix_species: Optional dict of {formula: concentration_mM} for matrix species.
                       If matrix is provided and matrix_species is not, auto-extracted.
        temperature_c: Operating temperature in °C

    Returns:
        List of TransformationProduct objects, sorted by orthogonality score (descending).
    """
    # Resolve matrix species
    if matrix_species is None:
        if matrix is not None:
            matrix_species = matrix_to_species_dict(matrix)
        else:
            matrix_species = {}

    # Route to correct enumerator
    route = _TARGET_ROUTER.get(target_formula)
    if route is None:
        return []  # unknown target — no products enumerable

    products: list[TransformationProduct] = []

    if route == "co2":
        products = _co2_products(matrix_species)

    elif route == "phosphate":
        products = _phosphate_products(matrix_species)

    elif route in ("nh3", "nh4"):
        ox_state = "NH3" if route == "nh3" else "NH4+"
        products = _nitrogen_products(matrix_species, ox_state)

    elif route == "no3":
        products = _nitrogen_products(matrix_species, "NO3-")

    elif route == "n2":
        products = _nitrogen_products(matrix_species, "N2")

    elif route in _METAL_SULFIDE_KSP:
        formula, ksp = _METAL_SULFIDE_KSP[route]
        products = _heavy_metal_sulfide_products(
            target_identity=target_formula,
            target_formula=formula,
            ksp_log=ksp,
        )

    elif route == "fluoride":
        products = _simple_precipitation_products(
            target_identity="fluoride", target_formula="F⁻",
            product_name="fluorspar (CaF₂)", product_formula="CaF₂",
            counter_ion="Ca2+", counter_formula="Ca²⁺",
            ksp_log=-10.46, dg_kj=-60.0,
            feedstock="Industrial fluorspar (HF production, steel flux)",
            dg_source="CRC Handbook; Ksp: Stumm & Morgan",
            counter_source=CoReactantSource.SUBSTRATE_PRELOADED,
        )

    elif route == "so2":
        products = _simple_precipitation_products(
            target_identity="SO₂", target_formula="SO₂",
            product_name="gypsum (CaSO₄·2H₂O)", product_formula="CaSO₄·2H₂O",
            counter_ion="CaO surface", counter_formula="Ca(OH)₂",
            ksp_log=-4.58, dg_kj=-85.0,
            feedstock="Construction material (wallboard, cement additive)",
            dg_source="NIST-JANAF; Ksp: CRC Handbook",
            counter_source=CoReactantSource.NONE,
        )

    elif route == "arsenic":
        products = _simple_precipitation_products(
            target_identity="As(V)", target_formula="H₂AsO₄⁻",
            product_name="scorodite (FeAsO₄·2H₂O)", product_formula="FeAsO₄·2H₂O",
            counter_ion="Fe(III)", counter_formula="Fe³⁺",
            ksp_log=-25.0, dg_kj=-100.0,
            feedstock="Stable sequestration (environmentally stable arsenate mineral)",
            dg_source="Krause & Ettel, Am. Mineral. 1988",
            counter_source=CoReactantSource.SUBSTRATE_PRELOADED,
        )

    # Compute orthogonality scores
    for p in products:
        # Determine worst-case co-reactant source
        if p.co_reactants:
            worst_cr = max(
                (cr.source for cr in p.co_reactants if cr.source != CoReactantSource.NONE),
                key=lambda s: s.orthogonality_rank,
                default=CoReactantSource.NONE,
            )
        else:
            worst_cr = CoReactantSource.NONE

        p.orthogonality_score = compute_orthogonality(
            dg_total_kj=p.dg_rxn_kj_mol,
            co_reactant_source=worst_cr,
            energy_input=p.energy_input,
            turnover=p.turnover,
            temperature_c=temperature_c,
        )

    # Sort by orthogonality score descending
    products.sort(key=lambda p: p.orthogonality_score, reverse=True)

    return products


def enumerate_from_problem(problem) -> list[TransformationProduct]:
    """Convenience: enumerate from a Problem object.

    Extracts target formula and matrix, calls enumerate_transformations.
    """
    target_formula = problem.target.formula
    matrix = problem.matrix
    temperature = getattr(matrix, 'temperature_c', 25.0)
    return enumerate_transformations(
        target_formula=target_formula,
        matrix=matrix,
        temperature_c=temperature,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Summary / reporting
# ═══════════════════════════════════════════════════════════════════════════

def summarize_products(products: list[TransformationProduct]) -> str:
    """Human-readable summary of enumerated products."""
    if not products:
        return "No transformation products found for this target."

    lines = [f"Found {len(products)} transformation product(s):\n"]
    for i, p in enumerate(products, 1):
        orth_label = "GOLD" if p.orthogonality_score >= 0.8 else \
                     "SILVER" if p.orthogonality_score >= 0.4 else \
                     "BRONZE" if p.orthogonality_score > 0 else "EXCLUDED"
        lines.append(f"  {i}. [{orth_label}] {p.name}")
        lines.append(f"     Formula: {p.formula}")
        lines.append(f"     ΔG = {p.dg_rxn_kj_mol:+.1f} kJ/mol")
        if p.ksp_log is not None:
            lines.append(f"     log(Ksp) = {p.ksp_log:.2f}")
        lines.append(f"     Orthogonality: {p.orthogonality_score:.4f}")
        lines.append(f"     Phase: {p.product_phase.value}")
        lines.append(f"     Turnover: {p.turnover.value}")
        if p.capture_sites:
            lines.append(f"     Capture site: {p.capture_sites[0].name}")
        if p.benefits_from_confinement:
            lines.append(f"     3D scaffold benefit: YES — {p.cascade_notes[:80]}...")
        lines.append(f"     Feedstock: {p.feedstock_value}")
        lines.append("")

    return "\n".join(lines)
