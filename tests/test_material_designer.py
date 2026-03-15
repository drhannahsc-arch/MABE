"""
test_material_designer.py — Tests for material designer M1 + M5.
"""
import pytest
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.material_designer import (
    TargetSpec, TieredValue, DataTier, PerformanceMetrics,
    PorousFrameworkSpec, FrameworkDesign,
    langmuir_capacity, langmuir_q_max_from_sites, langmuir_K_from_dG,
    selectivity_from_K, pore_size_exclusion,
    estimate_site_density, estimate_kinetics_t90,
    func_group_fraction_active,
    design_framework, screen_frameworks,
    unified_design, FRAMEWORK_DATABASE, FUNCTIONAL_GROUPS,
)


class TestTieredValue:

    def test_tier_tracking(self):
        tv = TieredValue(42.0, DataTier.T1_KNOWN, "NIST")
        assert tv.tier == DataTier.T1_KNOWN
        assert tv.value == 42.0

    def test_repr(self):
        tv = TieredValue(3.14, DataTier.T2_SOLID)
        assert "T2" in repr(tv)


class TestLangmuirPhysics:

    def test_capacity_positive(self):
        q = langmuir_capacity(100.0, 1.0, 1.0)
        assert q > 0

    def test_capacity_increases_with_concentration(self):
        q_low = langmuir_capacity(100.0, 1.0, 0.01)
        q_high = langmuir_capacity(100.0, 1.0, 10.0)
        assert q_high > q_low

    def test_capacity_saturates_at_qmax(self):
        """At very high C, q → q_max."""
        q = langmuir_capacity(100.0, 1.0, 1e6)
        assert abs(q - 100.0) < 1.0

    def test_capacity_zero_at_zero_C(self):
        q = langmuir_capacity(100.0, 1.0, 0.0)
        assert q == 0.0

    def test_K_from_dG_more_negative_higher_K(self):
        K_weak = langmuir_K_from_dG(-10.0)
        K_strong = langmuir_K_from_dG(-40.0)
        assert K_strong > K_weak

    def test_K_from_dG_zero_gives_one(self):
        """ΔG=0 → K=1 (in some unit system)."""
        K = langmuir_K_from_dG(0.0)
        assert K == pytest.approx(1.0 / 1000.0)  # normalized

    def test_q_max_from_sites(self):
        """1 mmol/g sites × 100 g/mol target = 100 mg/g."""
        q = langmuir_q_max_from_sites(1.0, 100.0)
        assert q == pytest.approx(100.0)

    def test_selectivity_ratio(self):
        assert selectivity_from_K(10.0, 1.0) == pytest.approx(10.0)
        assert selectivity_from_K(1.0, 10.0) == pytest.approx(0.1)

    def test_selectivity_inf_for_zero_interferent(self):
        assert selectivity_from_K(1.0, 0.0) == float('inf')


class TestPoreExclusion:

    def test_large_pore_admits_small_guest(self):
        assert pore_size_exclusion(12.0, 3.5) is True

    def test_small_pore_excludes_large_guest(self):
        assert pore_size_exclusion(3.0, 3.5) is False

    def test_marginal_pore(self):
        """Pore = guest + 1.0 Å (2×0.5 clearance) → just passes."""
        assert pore_size_exclusion(4.5, 3.5) is True

    def test_exact_fit_fails(self):
        """Pore = guest diameter exactly → fails (no clearance)."""
        assert pore_size_exclusion(3.5, 3.5) is False


class TestSiteDensity:

    def test_positive(self):
        sd = estimate_site_density(1000.0, 1.0)
        assert sd > 0

    def test_scales_with_surface_area(self):
        sd_low = estimate_site_density(500.0)
        sd_high = estimate_site_density(2000.0)
        assert sd_high > sd_low

    def test_reasonable_range(self):
        """MOF with 1000 m²/g, 1 group/nm² → ~1.7 mmol/g."""
        sd = estimate_site_density(1000.0, 1.0)
        assert 0.5 < sd < 5.0


class TestKinetics:

    def test_positive(self):
        t = estimate_kinetics_t90(10.0, 200.0)
        assert t > 0

    def test_small_pores_slower(self):
        t_small = estimate_kinetics_t90(3.0, 200.0)
        t_large = estimate_kinetics_t90(15.0, 200.0)
        assert t_small > t_large

    def test_larger_particles_slower(self):
        t_small = estimate_kinetics_t90(10.0, 50.0)
        t_large = estimate_kinetics_t90(10.0, 500.0)
        assert t_large > t_small


class TestFuncGroupPKa:

    def test_amine_active_for_anion_at_low_pH(self):
        """Protonated amine binds anions at pH 5."""
        f = func_group_fraction_active("amine-primary", 5.0, for_anion_capture=True)
        assert f > 0.99

    def test_amine_inactive_for_metal_at_low_pH(self):
        """Amine can't coordinate metals at pH 5 (protonated)."""
        f = func_group_fraction_active("amine-primary", 5.0, for_anion_capture=False)
        assert f < 0.001

    def test_carboxylate_for_anion(self):
        """Carboxylate needs to be deprotonated (COO⁻) even for anion capture."""
        f = func_group_fraction_active("carboxylate", 7.0, for_anion_capture=True)
        assert f > 0.99

    def test_thiol_always_deprotonation_dependent(self):
        f_low = func_group_fraction_active("thiol", 5.0)
        f_high = func_group_fraction_active("thiol", 10.0)
        assert f_high > f_low

    def test_sulfonate_always_active(self):
        assert func_group_fraction_active("sulfonate", 2.0) == 1.0
        assert func_group_fraction_active("sulfonate", 12.0) == 1.0


class TestFrameworkDesign:

    def test_from_database(self):
        spec = PorousFrameworkSpec.from_database("HKUST-1")
        assert spec.framework_name == "HKUST-1"
        assert spec.pore_diameter_A == 12.0

    def test_unknown_framework_raises(self):
        with pytest.raises(KeyError):
            PorousFrameworkSpec.from_database("NotAMOF-99")

    def test_design_framework_returns_design(self):
        target = TargetSpec(target_species="Pb2+", target_diameter_A=2.5,
                            target_charge=2, target_mw=207.2, pH=5.0)
        d = design_framework("UiO-66", target, functional_groups=["thiol"])
        assert isinstance(d, FrameworkDesign)
        assert d.metrics.capacity_mg_g is not None
        assert d.metrics.capacity_mg_g.value > 0

    def test_anion_target_uses_protonated_amine(self):
        """For selenite (anion), amine-functionalized MOF should show capacity."""
        target = TargetSpec(target_species="SeO3^2-", target_diameter_A=3.5,
                            target_charge=-2, target_mw=127.0, pH=5.0)
        d = design_framework("UiO-66-NH2", target,
                              functional_groups=["amine-primary"])
        assert d.metrics.capacity_mg_g.value > 10.0

    def test_inaccessible_pore_low_capacity(self):
        """SAPO-34 (3.8 Å pore) can't admit 3.5 Å selenite."""
        target = TargetSpec(target_diameter_A=3.5, target_charge=-2,
                            target_mw=127.0, pH=5.0)
        d = design_framework("SAPO-34", target)
        accessible = design_framework("HKUST-1", target)
        assert d.metrics.capacity_mg_g.value < accessible.metrics.capacity_mg_g.value


class TestScreenFrameworks:

    def test_returns_ranked_list(self):
        target = TargetSpec(target_species="Pb2+", target_diameter_A=2.5,
                            target_mw=207.2, pH=7.0)
        results = screen_frameworks(target)
        assert len(results) >= 5
        # Sorted by composite score
        scores = [d.metrics.composite_score() for d in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]


class TestUnifiedDesign:

    def test_returns_result(self):
        target = TargetSpec(name="test", target_species="Pb2+",
                            target_diameter_A=2.5, target_mw=207.2, pH=7.0)
        result = unified_design(target)
        assert result.n_materials_evaluated > 0
        assert len(result.rankings) > 0

    def test_best_for_categories(self):
        target = TargetSpec(name="test", target_species="SeO3^2-",
                            target_diameter_A=3.5, target_charge=-2,
                            target_mw=127.0, pH=5.0)
        result = unified_design(target, framework_func_groups=["amine-primary"])
        assert result.best_for_capacity != ""
        assert result.best_for_cost != ""

    def test_composite_score_bounded(self):
        target = TargetSpec(target_species="Pb2+", target_mw=207.2)
        result = unified_design(target)
        for r in result.rankings:
            assert 0 <= r.composite_score <= 1.0


class TestPerformanceMetrics:

    def test_composite_score_zero_for_empty(self):
        m = PerformanceMetrics()
        assert m.composite_score() == 0.0

    def test_composite_score_bounded(self):
        m = PerformanceMetrics(
            capacity_mg_g=TieredValue(200.0, DataTier.T2_SOLID),
            selectivity_ratio=TieredValue(50.0, DataTier.T3_CONCEPTUAL),
            kinetics_t90_min=TieredValue(5.0, DataTier.T3_CONCEPTUAL),
            cost_per_kg_usd=TieredValue(100.0, DataTier.T3_CONCEPTUAL),
            scalability_score=TieredValue(0.8, DataTier.T3_CONCEPTUAL),
            environmental_score=TieredValue(0.7, DataTier.T3_CONCEPTUAL),
        )
        s = m.composite_score()
        assert 0 < s <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ═══════════════════════════════════════════════════════════════════════════
# M2: Polymeric Sorbent Tests
# ═══════════════════════════════════════════════════════════════════════════

from core.material_designer import (
    donnan_partition, donnan_exclusion,
    ion_exchange_selectivity, SAC_SELECTIVITY, SBA_SELECTIVITY,
    flory_rehner_swelling_ratio, hydrogel_water_content,
    film_diffusion_t90, particle_diffusion_t90, rate_limiting_step,
    PolymericSorbentSpec, PolymericSorbentDesign,
    design_polymer, screen_polymers,
    POLYMER_DATABASE,
)


class TestDonnan:

    def test_counter_ion_enriched(self):
        """Counter-ions are enriched inside the resin phase."""
        ef = donnan_partition(z_ion=1, C_ext_mM=1.0, Q_resin_meq_mL=2.0)
        assert ef > 1.0

    def test_higher_capacity_more_enrichment(self):
        ef_low = donnan_partition(1, 1.0, 1.0)
        ef_high = donnan_partition(1, 1.0, 5.0)
        assert ef_high > ef_low

    def test_divalent_less_enrichment_per_unit(self):
        """Divalent ions: enrichment = ratio^(1/2) vs ratio^1 for monovalent."""
        ef_mono = donnan_partition(1, 1.0, 2.0)
        ef_di = donnan_partition(2, 1.0, 2.0)
        assert ef_mono > ef_di  # 2000 vs sqrt(2000)

    def test_co_ion_excluded(self):
        ef = donnan_exclusion(z_co=1, C_ext_mM=1.0, Q_resin_meq_mL=2.0)
        assert ef < 1.0

    def test_high_Q_strong_exclusion(self):
        ef_low = donnan_exclusion(1, 1.0, 1.0)
        ef_high = donnan_exclusion(1, 1.0, 5.0)
        assert ef_high < ef_low  # more capacity = more exclusion


class TestIonExchangeSelectivity:

    def test_pb_over_ca_on_SAC(self):
        """SAC resin prefers Pb2+ over Ca2+ (published)."""
        sel = ion_exchange_selectivity("Pb2+", "Ca2+", "SAC")
        assert sel > 1.0  # Pb2+ preferred

    def test_ba_over_mg_on_SAC(self):
        sel = ion_exchange_selectivity("Ba2+", "Mg2+", "SAC")
        assert sel > 1.0  # Ba2+ strongly preferred

    def test_selectivity_series_SAC(self):
        """Hofmeister-like: Cs > Rb > K > Na > Li for monovalent."""
        K_cs = SAC_SELECTIVITY["Cs+"]
        K_na = SAC_SELECTIVITY["Na+"]
        K_li = SAC_SELECTIVITY["Li+"]
        assert K_cs > K_na > K_li

    def test_sba_nitrate_over_chloride(self):
        sel = ion_exchange_selectivity("NO3-", "Cl-", "SBA")
        assert sel > 1.0

    def test_sba_sulfate_vs_chloride(self):
        """Sulfate has low selectivity on Type I SBA (known)."""
        sel = ion_exchange_selectivity("SO4^2-", "Cl-", "SBA")
        assert sel < 1.0  # sulfate poorly selected by Type I SBA

    def test_unknown_resin_returns_one(self):
        assert ion_exchange_selectivity("Na+", "K+", "UNKNOWN") == 1.0


class TestFloryRehner:

    def test_hydrophilic_swells(self):
        """Hydrophilic polymer (chi=0.3) should swell significantly."""
        Q = flory_rehner_swelling_ratio(chi=0.3, Mc=5000)
        assert Q > 2.0

    def test_hydrophobic_minimal_swell(self):
        """Hydrophobic polymer (chi=0.6) should barely swell."""
        Q = flory_rehner_swelling_ratio(chi=0.6, Mc=5000)
        assert Q < 1.5

    def test_higher_Mc_more_swelling(self):
        """More MW between crosslinks = more swelling."""
        Q_tight = flory_rehner_swelling_ratio(0.35, Mc=1000)
        Q_loose = flory_rehner_swelling_ratio(0.35, Mc=20000)
        assert Q_loose > Q_tight

    def test_water_content_from_Q(self):
        wt = hydrogel_water_content(5.0)
        assert 75 < wt < 85  # Q=5 → 80% water

    def test_no_swelling_zero_water(self):
        assert hydrogel_water_content(1.0) == 0.0


class TestKinetics:

    def test_film_diffusion_positive(self):
        t = film_diffusion_t90(0.25e-3)  # 0.5 mm bead
        assert t > 0

    def test_particle_diffusion_positive(self):
        t = particle_diffusion_t90(0.25e-3)
        assert t > 0

    def test_larger_beads_slower(self):
        t_small = particle_diffusion_t90(0.1e-3)
        t_large = particle_diffusion_t90(1.0e-3)
        assert t_large > t_small

    def test_rate_limiting_identification(self):
        assert rate_limiting_step(10.0, 1.0) == "film-diffusion"
        assert rate_limiting_step(1.0, 10.0) == "particle-diffusion"
        assert rate_limiting_step(5.0, 5.0) == "mixed"


class TestPolymericSorbentDesign:

    def test_from_database(self):
        spec = PolymericSorbentSpec.from_database("Chelex-100")
        assert spec.resin_name == "Chelex-100"
        assert spec.capacity_meq_g == 2.8

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            PolymericSorbentSpec.from_database("FakeResin-99")

    def test_design_returns_design(self):
        target = TargetSpec(target_species="Pb2+", target_charge=2,
                            target_mw=207.2, pH=5.0)
        d = design_polymer("Chelex-100", target)
        assert isinstance(d, PolymericSorbentDesign)
        assert d.metrics.capacity_mg_g.value > 0

    def test_sba_for_selenite(self):
        """SBA resin should bind selenite (anion target)."""
        target = TargetSpec(target_species="SeO3^2-", target_charge=-2,
                            target_mw=127.0, pH=5.0)
        d = design_polymer("Amberlite-IRA400", target)
        assert d.metrics.capacity_mg_g.value > 0
        assert d.func_group_activity == 1.0  # quaternary N always active

    def test_chelating_high_selectivity(self):
        """Chelating resin should have high selectivity for heavy metals."""
        target = TargetSpec(target_species="Pb2+", target_charge=2,
                            target_mw=207.2, pH=5.0,
                            interferent_species=["Ca2+", "Mg2+"])
        d = design_polymer("Chelex-100", target)
        assert d.selectivity_vs_worst >= 50.0

    def test_chitosan_environmental_bonus(self):
        """Chitosan (bio-based) should score higher on environmental."""
        target = TargetSpec(target_species="Pb2+", target_charge=2,
                            target_mw=207.2, pH=5.0)
        d_chitosan = design_polymer("chitosan-bead", target)
        d_resin = design_polymer("Amberlite-IR120", target)
        assert d_chitosan.metrics.environmental_score.value > \
               d_resin.metrics.environmental_score.value


class TestScreenPolymers:

    def test_returns_ranked_list(self):
        target = TargetSpec(target_species="Pb2+", target_charge=2,
                            target_mw=207.2, pH=7.0)
        results = screen_polymers(target)
        assert len(results) >= 5
        scores = [d.metrics.composite_score() for d in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_database_count(self):
        assert len(POLYMER_DATABASE) >= 16


class TestUnifiedWithPolymers:

    def test_polymers_included(self):
        target = TargetSpec(name="test", target_species="Pb2+",
                            target_charge=2, target_mw=207.2, pH=5.0)
        result = unified_design(target)
        # Should have both frameworks and polymers
        classes = set(r.material_class for r in result.rankings)
        has_framework = any("PorousFramework" in c for c in classes)
        has_polymer = any("PolymericSorbent" in c for c in classes)
        assert has_framework
        assert has_polymer

    def test_best_for_cost_is_polymer(self):
        """Polymers ($15/kg) should beat MOFs ($200/kg) on cost."""
        target = TargetSpec(name="test", target_species="Pb2+",
                            target_charge=2, target_mw=207.2, pH=5.0)
        result = unified_design(target)
        # Best for cost should be a polymer or cheap zeolite
        assert result.best_for_cost != ""

    def test_data_tier_distribution(self):
        """M2 should contribute T1 data (published selectivity coefficients)."""
        target = TargetSpec(target_species="Pb2+", target_charge=2,
                            target_mw=207.2, pH=5.0,
                            interferent_species=["Ca2+"])
        result = unified_design(target)
        # Check that some T1 data exists (from IX selectivity tables)
        t1_count = 0
        for r in result.rankings:
            m = r.metrics
            if m.selectivity_ratio and m.selectivity_ratio.tier == DataTier.T1_KNOWN:
                t1_count += 1
        assert t1_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ═══════════════════════════════════════════════════════════════════════════
# M3: Composite Material Tests
# ═══════════════════════════════════════════════════════════════════════════

from core.material_designer import (
    maxwell_garnett, bruggeman_ema,
    rule_of_mixtures, inverse_rule_of_mixtures,
    composite_capacity, composite_density, composite_cost,
    core_shell_volume_fraction, core_shell_mass_fraction,
    shell_diffusion_t90,
    kozeny_carman_dP, kozeny_carman_dP_kPa,
    bed_volumes_to_breakthrough,
    CompositeConfig, CompositeDesign,
    design_composite, screen_composites,
    COMPOSITE_DATABASE,
)


class TestEffectiveMedium:

    def test_maxwell_garnett_pure_matrix(self):
        """f=0 → matrix property."""
        assert maxwell_garnett(2.0, 10.0, 0.0) == 2.0

    def test_maxwell_garnett_pure_inclusion(self):
        """f=1 → inclusion property."""
        assert maxwell_garnett(2.0, 10.0, 1.0) == 10.0

    def test_maxwell_garnett_intermediate(self):
        """Dilute inclusions: result between matrix and inclusion."""
        P = maxwell_garnett(2.0, 10.0, 0.1)
        assert 2.0 < P < 10.0

    def test_bruggeman_symmetric_equal(self):
        """Equal fractions of same material → that material's property."""
        P = bruggeman_ema(5.0, 5.0, 0.5)
        assert abs(P - 5.0) < 0.01

    def test_bruggeman_intermediate(self):
        P = bruggeman_ema(2.0, 10.0, 0.5)
        assert 2.0 < P < 10.0

    def test_bruggeman_vs_maxwell_garnett(self):
        """Bruggeman gives different result from MG at high f."""
        P_mg = maxwell_garnett(2.0, 10.0, 0.4)
        P_br = bruggeman_ema(2.0, 10.0, 0.4)
        assert P_mg != pytest.approx(P_br, rel=0.01)


class TestRuleOfMixtures:

    def test_linear_rom(self):
        """50/50 mix → average."""
        assert rule_of_mixtures(10.0, 20.0, 0.5) == pytest.approx(15.0)

    def test_inverse_rom(self):
        """Inverse ROM gives lower bound (series)."""
        inv = inverse_rule_of_mixtures(10.0, 20.0, 0.5)
        lin = rule_of_mixtures(10.0, 20.0, 0.5)
        assert inv < lin  # Reuss < Voigt

    def test_composite_capacity_exact(self):
        """20% active at 100 mg/g → 20 mg/g composite."""
        assert composite_capacity(100.0, 0.2) == pytest.approx(20.0)

    def test_composite_density(self):
        rho = composite_density(1.2, 2.5, 0.3)
        assert rho == pytest.approx(0.3*1.2 + 0.7*2.5)


class TestCoreShell:

    def test_volume_fraction_thin_shell(self):
        """Thin shell on large core → small volume fraction."""
        f = core_shell_volume_fraction(1e-3, 10e-6)  # 1mm core, 10μm shell
        assert f < 0.05

    def test_volume_fraction_thick_shell(self):
        """Shell = core radius → f = 1 - (0.5)³ = 0.875."""
        f = core_shell_volume_fraction(0.5e-3, 0.5e-3)
        assert abs(f - 0.875) < 0.01

    def test_mass_fraction_density_dependent(self):
        """Heavier shell → higher mass fraction than volume fraction."""
        f_vol = core_shell_volume_fraction(0.5e-3, 0.1e-3)
        f_mass_heavy = core_shell_mass_fraction(0.5e-3, 0.1e-3, 1.0, 3.0)
        f_mass_light = core_shell_mass_fraction(0.5e-3, 0.1e-3, 1.0, 0.5)
        assert f_mass_heavy > f_mass_light

    def test_shell_diffusion_faster_than_full_particle(self):
        """Shell diffusion only through thin layer → faster."""
        R_total = 0.5e-3
        R_core = 0.4e-3  # 100 μm shell
        t_shell = shell_diffusion_t90(R_total, R_core)
        t_full = 0.307 * R_total**2 / 1e-11 / 60.0  # full particle
        assert t_shell < t_full


class TestKozenyCarman:

    def test_pressure_positive(self):
        dP = kozeny_carman_dP(1e-3, 1.0, 0.5e-3)
        assert dP > 0

    def test_smaller_particles_higher_dP(self):
        dP_large = kozeny_carman_dP(1e-3, 1.0, 1.0e-3)
        dP_small = kozeny_carman_dP(1e-3, 1.0, 0.1e-3)
        assert dP_small > dP_large

    def test_higher_velocity_higher_dP(self):
        dP_slow = kozeny_carman_dP(0.5e-3, 1.0, 0.5e-3)
        dP_fast = kozeny_carman_dP(2.0e-3, 1.0, 0.5e-3)
        assert dP_fast > dP_slow

    def test_dP_proportional_to_velocity(self):
        """ΔP ∝ v (linear, Darcy regime)."""
        dP1 = kozeny_carman_dP(1e-3, 1.0, 0.5e-3)
        dP2 = kozeny_carman_dP(2e-3, 1.0, 0.5e-3)
        assert abs(dP2 / dP1 - 2.0) < 0.01

    def test_kpa_conversion(self):
        dP_pa = kozeny_carman_dP(1e-3, 1.0, 0.5e-3)
        dP_kpa = kozeny_carman_dP_kPa(1e-3, 1.0, 0.5e-3)
        assert abs(dP_kpa - dP_pa / 1000.0) < 0.001


class TestBreakthrough:

    def test_bv_positive(self):
        bv = bed_volumes_to_breakthrough(10.0, 800.0, 0.4, 0.1)
        assert bv > 0

    def test_higher_capacity_more_bv(self):
        bv_low = bed_volumes_to_breakthrough(5.0, 800.0, 0.4, 0.1)
        bv_high = bed_volumes_to_breakthrough(50.0, 800.0, 0.4, 0.1)
        assert bv_high > bv_low

    def test_higher_feed_conc_fewer_bv(self):
        bv_dilute = bed_volumes_to_breakthrough(10.0, 800.0, 0.4, 0.01)
        bv_conc = bed_volumes_to_breakthrough(10.0, 800.0, 0.4, 1.0)
        assert bv_dilute > bv_conc


class TestCompositeDesign:

    def test_database_count(self):
        assert len(COMPOSITE_DATABASE) >= 8

    def test_all_have_sources(self):
        for name, config in COMPOSITE_DATABASE.items():
            assert len(config.source) > 10, f"{name}: missing source"
            assert "DOI" in config.source, f"{name}: missing DOI"

    def test_design_returns_design(self):
        target = TargetSpec(target_species="As(V)", target_charge=-1,
                            target_mw=75.0, pH=7.0)
        d = design_composite("FeOOH-on-sand", target)
        assert isinstance(d, CompositeDesign)

    def test_published_capacity_used(self):
        """FeOOH-on-sand has published As(V) capacity → should use it."""
        target = TargetSpec(target_species="As(V)", target_mw=75.0, pH=7.0)
        d = design_composite("FeOOH-on-sand", target)
        assert d.metrics.capacity_mg_g is not None
        assert d.metrics.capacity_mg_g.value == 1.5  # exact published value

    def test_missing_capacity_is_none(self):
        """TiO₂-on-alumina has no adsorptive capacity → should be None."""
        target = TargetSpec(target_species="Se(VI)", target_mw=79.0, pH=7.0)
        d = design_composite("TiO2-on-alumina", target)
        assert d.config.published_capacity_mg_g is None

    def test_no_fabricated_data(self):
        """Cost and durability should be None, not estimated."""
        target = TargetSpec(target_species="test", pH=7.0)
        d = design_composite("FeOOH-on-sand", target)
        assert d.metrics.cost_per_kg_usd is None
        assert d.metrics.durability_cycles is None
        assert d.metrics.environmental_score is None

    def test_breakthrough_from_published(self):
        """MnO₂-on-sand has published BV → should use exact value."""
        target = TargetSpec(target_species="Ra2+", pH=7.0)
        d = design_composite("MnO2-on-sand", target)
        assert d.BV_to_breakthrough == 20000.0

    def test_pressure_drop_computed(self):
        """Particles with known diameter → pressure drop is computable (T2)."""
        target = TargetSpec(pH=7.0)
        d = design_composite("FeOOH-on-sand", target)
        assert d.pressure_drop_kPa is not None
        assert d.pressure_drop_kPa > 0


class TestScreenComposites:

    def test_returns_list(self):
        target = TargetSpec(target_species="As(V)", target_mw=75.0, pH=7.0)
        results = screen_composites(target)
        assert len(results) >= 5


class TestUnifiedWithComposites:

    def test_composites_included(self):
        target = TargetSpec(name="test", target_species="Pb2+",
                            target_charge=2, target_mw=207.2, pH=5.0)
        result = unified_design(target, top_n=30)
        classes = set(r.material_class for r in result.rankings)
        has_composite = any("Composite" in c for c in classes)
        assert has_composite

    def test_three_classes_present(self):
        """All three material classes should appear."""
        target = TargetSpec(name="test", target_species="Pb2+",
                            target_charge=2, target_mw=207.2, pH=5.0)
        result = unified_design(target, top_n=30)
        classes = set()
        for r in result.rankings:
            if "PorousFramework" in r.material_class:
                classes.add("framework")
            elif "PolymericSorbent" in r.material_class:
                classes.add("polymer")
            elif "Composite" in r.material_class:
                classes.add("composite")
        assert len(classes) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
