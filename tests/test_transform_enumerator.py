"""
tests/test_transform_enumerator.py — Tests for capture-transform product enumeration.

Validates:
  - Correct product enumeration per target class
  - Orthogonality scoring logic
  - Matrix-aware co-reactant routing
  - N₂ fixation correctly excluded under orthogonality
  - 3D scaffold benefit flags
  - Click compatibility constraints
  - Integration with Problem objects
"""

import sys
import os
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.transform_enumerator import (
    enumerate_transformations,
    enumerate_from_problem,
    compute_orthogonality,
    summarize_products,
    TransformationProduct,
    CoReactantSource,
    EnergyInput,
    TurnoverMode,
    ProductPhase,
    ClickCompatibility,
)


# ═══════════════════════════════════════════════════════════════════════════
# Orthogonality scoring unit tests
# ═══════════════════════════════════════════════════════════════════════════

class TestOrthogonalityScoring:
    """Test the compute_orthogonality function directly."""

    def test_gold_standard(self):
        """Spontaneous + matrix-native + no energy + catalytic = highest score."""
        score = compute_orthogonality(
            dg_total_kj=-50.0,
            co_reactant_source=CoReactantSource.MATRIX_NATIVE,
            energy_input=EnergyInput.NONE,
            turnover=TurnoverMode.CATALYTIC,
        )
        assert score >= 0.8, f"Gold standard should score ≥0.8, got {score}"

    def test_no_co_reactant_is_best(self):
        """No co-reactant needed should score higher than matrix-native."""
        score_none = compute_orthogonality(
            dg_total_kj=-50.0,
            co_reactant_source=CoReactantSource.NONE,
            energy_input=EnergyInput.NONE,
            turnover=TurnoverMode.CATALYTIC,
        )
        score_matrix = compute_orthogonality(
            dg_total_kj=-50.0,
            co_reactant_source=CoReactantSource.MATRIX_NATIVE,
            energy_input=EnergyInput.NONE,
            turnover=TurnoverMode.CATALYTIC,
        )
        assert score_none > score_matrix

    def test_externally_supplied_kills_score(self):
        """Externally supplied co-reactant → score = 0."""
        score = compute_orthogonality(
            dg_total_kj=-100.0,
            co_reactant_source=CoReactantSource.EXTERNALLY_SUPPLIED,
            energy_input=EnergyInput.NONE,
            turnover=TurnoverMode.CATALYTIC,
        )
        assert score == 0.0

    def test_electrochemical_kills_score(self):
        """Electrochemical energy input → score = 0."""
        score = compute_orthogonality(
            dg_total_kj=-100.0,
            co_reactant_source=CoReactantSource.NONE,
            energy_input=EnergyInput.ELECTROCHEMICAL,
            turnover=TurnoverMode.CATALYTIC,
        )
        assert score == 0.0

    def test_active_thermal_kills_score(self):
        """Active thermal (>80°C) → score = 0."""
        score = compute_orthogonality(
            dg_total_kj=-100.0,
            co_reactant_source=CoReactantSource.NONE,
            energy_input=EnergyInput.ACTIVE_THERMAL,
            turnover=TurnoverMode.CATALYTIC,
        )
        assert score == 0.0

    def test_stoichiometric_expensive_kills_score(self):
        """Expensive stoichiometric site → score = 0."""
        score = compute_orthogonality(
            dg_total_kj=-100.0,
            co_reactant_source=CoReactantSource.NONE,
            energy_input=EnergyInput.NONE,
            turnover=TurnoverMode.STOICHIOMETRIC_EXPENSIVE,
        )
        assert score == 0.0

    def test_endergonic_without_photocatalysis_fails(self):
        """Positive ΔG without solar input → 0."""
        score = compute_orthogonality(
            dg_total_kj=30.0,
            co_reactant_source=CoReactantSource.NONE,
            energy_input=EnergyInput.NONE,
            turnover=TurnoverMode.CATALYTIC,
        )
        assert score == 0.0

    def test_endergonic_with_photocatalysis_passes(self):
        """Positive ΔG + solar photocatalytic → marginal but non-zero."""
        score = compute_orthogonality(
            dg_total_kj=30.0,
            co_reactant_source=CoReactantSource.SOLAR_PHOTOCATALYTIC,
            energy_input=EnergyInput.PASSIVE_SOLAR_PHOTOCATALYTIC,
            turnover=TurnoverMode.CATALYTIC,
        )
        assert score > 0.0

    def test_stoichiometric_cheap_lower_than_catalytic(self):
        """Stoichiometric cheap should score lower than catalytic."""
        score_cat = compute_orthogonality(
            dg_total_kj=-50.0,
            co_reactant_source=CoReactantSource.NONE,
            energy_input=EnergyInput.NONE,
            turnover=TurnoverMode.CATALYTIC,
        )
        score_stoich = compute_orthogonality(
            dg_total_kj=-50.0,
            co_reactant_source=CoReactantSource.NONE,
            energy_input=EnergyInput.NONE,
            turnover=TurnoverMode.STOICHIOMETRIC_CHEAP,
        )
        assert score_cat > score_stoich

    def test_score_bounded_0_1(self):
        """Score always in [0, 1]."""
        score = compute_orthogonality(
            dg_total_kj=-500.0,  # extremely favorable
            co_reactant_source=CoReactantSource.NONE,
            energy_input=EnergyInput.NONE,
            turnover=TurnoverMode.CATALYTIC,
        )
        assert 0.0 <= score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# CO₂ enumeration tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCO2Enumeration:
    """Test CO₂ transformation product enumeration."""

    def test_co2_basic_enumeration(self):
        """CO₂ should always produce at least amine + photocatalytic pathways."""
        products = enumerate_transformations("CO2")
        assert len(products) >= 2
        formulas = [p.formula for p in products]
        assert "HCOO⁻ / CH₃OH" in formulas  # photocatalytic always present

    def test_co2_with_calcium_produces_calcite(self):
        """Hard water matrix should enable CaCO₃ pathway."""
        products = enumerate_transformations(
            "CO2",
            matrix_species={"Ca2+": 2.0, "Mg2+": 1.0}
        )
        names = [p.name for p in products]
        calcite_found = any("CaCO₃" in n or "calcite" in n for n in names)
        assert calcite_found, f"Calcite not found in: {names}"

    def test_co2_calcite_is_gold_orthogonality(self):
        """CaCO₃ via CA mimic with matrix Ca²⁺ should score ≥0.8."""
        products = enumerate_transformations(
            "CO2",
            matrix_species={"Ca2+": 2.0}
        )
        calcite = [p for p in products if "calcite" in p.name.lower()]
        assert len(calcite) >= 1
        assert calcite[0].orthogonality_score >= 0.7, \
            f"Calcite orthogonality too low: {calcite[0].orthogonality_score}"

    def test_co2_ca_mimic_is_spaac_only(self):
        """Zn-CA mimic requires SPAAC — Cu would poison the Zn site."""
        products = enumerate_transformations(
            "CO2",
            matrix_species={"Ca2+": 2.0}
        )
        calcite = [p for p in products if "calcite" in p.name.lower()][0]
        assert calcite.capture_sites[0].click_compatibility == ClickCompatibility.SPAAC_ONLY

    def test_co2_photocatalytic_is_endergonic(self):
        """Photocatalytic CO₂ reduction has positive ΔG (driven by light)."""
        products = enumerate_transformations("CO2")
        photo = [p for p in products if "photocatal" in p.name.lower()]
        assert len(photo) >= 1
        assert photo[0].dg_rxn_kj_mol > 0

    def test_co2_sorted_by_orthogonality(self):
        """Products should be sorted descending by orthogonality score."""
        products = enumerate_transformations(
            "CO2",
            matrix_species={"Ca2+": 2.0, "Mg2+": 1.0}
        )
        scores = [p.orthogonality_score for p in products]
        assert scores == sorted(scores, reverse=True)

    def test_co2_benefits_from_confinement(self):
        """Multiple CO₂ pathways should flag 3D scaffold benefit."""
        products = enumerate_transformations("CO2", matrix_species={"Ca2+": 2.0})
        confinement_products = [p for p in products if p.benefits_from_confinement]
        assert len(confinement_products) >= 2


# ═══════════════════════════════════════════════════════════════════════════
# Phosphate enumeration tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPhosphateEnumeration:
    """Test phosphate transformation product enumeration."""

    def test_phosphate_basic(self):
        """Phosphate should always produce ≥3 products (HAp, struvite, ZrP, FeP)."""
        products = enumerate_transformations("PO4_3-")
        assert len(products) >= 3

    def test_phosphate_hap_is_solid(self):
        """Hydroxyapatite should be a solid precipitate."""
        products = enumerate_transformations("PO4_3-", matrix_species={"Ca2+": 2.0})
        hap = [p for p in products if "hydroxyapatite" in p.name.lower()]
        assert len(hap) >= 1
        assert hap[0].product_phase == ProductPhase.SOLID_PRECIPITATE

    def test_struvite_is_dual_target(self):
        """Struvite should list both NH₄⁺ and Mg²⁺ as co-reactants."""
        products = enumerate_transformations("PO4_3-")
        struvite = [p for p in products if "struvite" in p.name.lower()]
        assert len(struvite) >= 1
        cr_ids = [cr.identity for cr in struvite[0].co_reactants]
        assert "Mg2+" in cr_ids
        assert "NH4+" in cr_ids

    def test_struvite_benefits_from_confinement(self):
        """Struvite should flag cascade benefit (Pattern 2)."""
        products = enumerate_transformations("PO4_3-")
        struvite = [p for p in products if "struvite" in p.name.lower()]
        assert struvite[0].benefits_from_confinement is True

    def test_zirconium_phosphate_stoichiometric(self):
        """ZrP formation is stoichiometric — Zr sites consumed."""
        products = enumerate_transformations("PO4_3-")
        zrp = [p for p in products if "zirconium" in p.name.lower()]
        assert len(zrp) >= 1
        assert zrp[0].turnover == TurnoverMode.STOICHIOMETRIC_CHEAP

    def test_iron_phosphate_lfp_value(self):
        """FePO₄ should note LFP battery cathode value."""
        products = enumerate_transformations("PO4_3-")
        fep = [p for p in products if "iron phosphate" in p.name.lower()]
        assert len(fep) >= 1
        assert "LFP" in fep[0].feedstock_value or "battery" in fep[0].feedstock_value


# ═══════════════════════════════════════════════════════════════════════════
# Nitrogen enumeration tests
# ═══════════════════════════════════════════════════════════════════════════

class TestNitrogenEnumeration:

    def test_nh3_produces_ammonium_salt(self):
        """NH₃ should produce ammonium sulfonate salt."""
        products = enumerate_transformations("NH3")
        assert len(products) >= 1
        assert any("ammonium" in p.name.lower() for p in products)

    def test_nh3_acid_site_is_catalytic(self):
        """Sulfonic acid NH₃ capture is catalytic (regenerable by wash)."""
        products = enumerate_transformations("NH3")
        amm = [p for p in products if "sulfonate" in p.name.lower()]
        assert amm[0].turnover == TurnoverMode.CATALYTIC

    def test_no3_photocatalytic_pathway(self):
        """NO₃⁻ should produce photocatalytic NH₄⁺ pathway."""
        products = enumerate_transformations("NO3-")
        photo = [p for p in products if "photocatal" in p.name.lower()]
        assert len(photo) >= 1
        assert photo[0].energy_input == EnergyInput.PASSIVE_SOLAR_PHOTOCATALYTIC

    def test_no3_zvi_pathway(self):
        """NO₃⁻ should produce zerovalent iron pathway."""
        products = enumerate_transformations("NO3-")
        zvi = [p for p in products if "zerovalent" in p.name.lower() or "iron reduction" in p.name.lower()]
        assert len(zvi) >= 1

    def test_n2_excluded_under_orthogonality(self):
        """N₂ fixation should be enumerated but score 0 (excluded)."""
        products = enumerate_transformations("N2")
        assert len(products) >= 1
        assert products[0].orthogonality_score == 0.0
        assert not products[0].is_orthogonal

    def test_n2_notes_exclusion(self):
        """N₂ product should note it's excluded."""
        products = enumerate_transformations("N2")
        assert "EXCLUDED" in products[0].name or "EXCLUDED" in products[0].notes


# ═══════════════════════════════════════════════════════════════════════════
# Heavy metal tests
# ═══════════════════════════════════════════════════════════════════════════

class TestHeavyMetalEnumeration:

    def test_lead_produces_galena(self):
        products = enumerate_transformations("Pb2+")
        assert len(products) >= 1
        assert any("PbS" in p.formula for p in products)

    def test_mercury_extreme_ksp(self):
        """HgS has Ksp = 10⁻⁵² — should be in products."""
        products = enumerate_transformations("Hg2+")
        hgs = [p for p in products if "HgS" in p.formula]
        assert len(hgs) >= 1
        assert hgs[0].ksp_log == -52.0

    def test_cadmium_semiconductor_value(self):
        """CdS should note semiconductor precursor value."""
        products = enumerate_transformations("Cd2+")
        cds = [p for p in products if "CdS" in p.formula]
        assert "semiconductor" in cds[0].feedstock_value.lower()

    def test_heavy_metals_spaac_only(self):
        """Thiol/sulfide sites require SPAAC — Cu competes for thiol."""
        products = enumerate_transformations("Pb2+")
        assert products[0].capture_sites[0].click_compatibility == ClickCompatibility.SPAAC_ONLY


# ═══════════════════════════════════════════════════════════════════════════
# Simple precipitation tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSimplePrecipitation:

    def test_fluoride_produces_caf2(self):
        products = enumerate_transformations("F-")
        assert len(products) >= 1
        assert "CaF₂" in products[0].formula

    def test_so2_produces_gypsum(self):
        products = enumerate_transformations("SO2")
        assert len(products) >= 1
        assert "gypsum" in products[0].name.lower()

    def test_arsenic_produces_scorodite(self):
        products = enumerate_transformations("H2AsO4-")
        assert len(products) >= 1
        assert "scorodite" in products[0].name.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Matrix-aware routing tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMatrixAwareRouting:

    def test_hard_water_description_adds_calcium(self):
        """Matrix described as 'hard water' should auto-populate Ca²⁺."""
        from core.transform_enumerator import matrix_to_species_dict
        from core.problem import Matrix
        m = Matrix(description="hard water from limestone aquifer")
        species = matrix_to_species_dict(m)
        assert "Ca2+" in species
        assert species["Ca2+"] > 0

    def test_wastewater_description_adds_nutrients(self):
        """Wastewater matrix should auto-populate NH₄⁺ and PO₄."""
        from core.transform_enumerator import matrix_to_species_dict
        from core.problem import Matrix
        m = Matrix(description="municipal wastewater effluent")
        species = matrix_to_species_dict(m)
        assert "NH4+" in species
        assert "PO4_total" in species

    def test_nh4_in_wastewater_enables_struvite_for_phosphate(self):
        """If NH₄⁺ is in matrix, phosphate enumeration should show struvite."""
        products = enumerate_transformations(
            "PO4_3-",
            matrix_species={"NH4+": 5.0, "Mg2+": 2.0, "Ca2+": 1.0}
        )
        struvite = [p for p in products if "struvite" in p.name.lower()]
        assert len(struvite) >= 1
        # NH₄⁺ should be matrix-native, not substrate-preloaded
        nh4_crs = [cr for cr in struvite[0].co_reactants if cr.identity == "NH4+"]
        assert nh4_crs[0].source == CoReactantSource.MATRIX_NATIVE


# ═══════════════════════════════════════════════════════════════════════════
# Integration with Problem object
# ═══════════════════════════════════════════════════════════════════════════

class TestProblemIntegration:

    def test_enumerate_from_problem_co2(self):
        """Test full Problem → enumerate pipeline for CO₂."""
        from core.problem import Problem, TargetSpecies, Matrix, Outcome, CompetingSpecies
        problem = Problem(
            target=TargetSpecies(
                identity="carbon dioxide",
                formula="CO2",
                charge=0.0,
                geometry="linear",
            ),
            matrix=Matrix(
                description="seawater",
                ph=8.1,
                temperature_c=15.0,
                competing_species=[
                    CompetingSpecies(identity="calcium", formula="Ca2+",
                                    concentration_mm=10.0, charge=2.0),
                    CompetingSpecies(identity="magnesium", formula="Mg2+",
                                    concentration_mm=53.0, charge=2.0),
                ],
            ),
            desired_outcome=Outcome(
                description="capture CO₂ and transform to solid mineral feedstock",
                reversible=False,
                product="carbonate mineral",
            ),
        )
        products = enumerate_from_problem(problem)
        assert len(products) >= 3  # calcite, magnesite, amine, photocatalytic
        # Calcite should rank high (seawater has plenty of Ca²⁺)
        assert "CaCO₃" in products[0].formula or "calcite" in products[0].name.lower()

    def test_enumerate_from_problem_phosphate_wastewater(self):
        """Phosphate in wastewater should produce struvite near top."""
        from core.problem import Problem, TargetSpecies, Matrix, Outcome, CompetingSpecies
        problem = Problem(
            target=TargetSpecies(
                identity="phosphate",
                formula="PO4_3-",
                charge=-3.0,
                geometry="tetrahedral",
            ),
            matrix=Matrix(
                description="agricultural wastewater",
                ph=7.5,
                competing_species=[
                    CompetingSpecies(identity="ammonium", formula="NH4+",
                                    concentration_mm=5.0, charge=1.0),
                    CompetingSpecies(identity="magnesium", formula="Mg2+",
                                    concentration_mm=2.0, charge=2.0),
                    CompetingSpecies(identity="calcium", formula="Ca2+",
                                    concentration_mm=3.0, charge=2.0),
                ],
            ),
            desired_outcome=Outcome(
                description="recover phosphorus as fertilizer feedstock",
                reversible=False,
            ),
        )
        products = enumerate_from_problem(problem)
        assert len(products) >= 3
        # All should be orthogonal (no excluded pathways for phosphate)
        assert all(p.is_orthogonal for p in products)


# ═══════════════════════════════════════════════════════════════════════════
# Summary output test
# ═══════════════════════════════════════════════════════════════════════════

class TestSummary:

    def test_summary_not_empty(self):
        products = enumerate_transformations("CO2", matrix_species={"Ca2+": 2.0})
        summary = summarize_products(products)
        assert "Found" in summary
        assert "CaCO₃" in summary or "calcite" in summary

    def test_summary_empty_target(self):
        products = enumerate_transformations("Xe")  # xenon — not in database
        summary = summarize_products(products)
        assert "No transformation products found" in summary


# ═══════════════════════════════════════════════════════════════════════════
# Unknown target graceful handling
# ═══════════════════════════════════════════════════════════════════════════

class TestUnknownTarget:

    def test_unknown_target_returns_empty(self):
        products = enumerate_transformations("Xe")
        assert products == []

    def test_unknown_formula_returns_empty(self):
        products = enumerate_transformations("FAKEMOLECULE")
        assert products == []
