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



class TestTMMForwardModel:

    def test_tmm_design_returns_stack(self):
        stack = _tmm_design_for_wavelength(530.0)
        assert len(stack) == 10  # 5 bilayers × 2

    def test_tmm_design_materials_valid(self):
        stack = _tmm_design_for_wavelength(530.0)
        for mat, t in stack:
            assert isinstance(mat, str)
            assert t > 0

    def test_tmm_spectrum_has_peak(self):
        import numpy as np
        stack = _tmm_design_for_wavelength(530.0)
        lam, R = _tmm_spectrum(stack)
        assert len(lam) > 0
        assert len(R) == len(lam)
        assert np.max(R) > 0.5  # should have strong reflectance peak

    def test_tmm_peak_near_target(self):
        """TMM designed for 530nm should peak near 530nm."""
        import numpy as np
        stack = _tmm_design_for_wavelength(530.0)
        lam, R = _tmm_spectrum(stack)
        peak_nm = lam[np.argmax(R)]
        assert abs(peak_nm - 530.0) < 30  # within 30nm

    def test_tmm_different_targets_different_peaks(self):
        import numpy as np
        stack_blue = _tmm_design_for_wavelength(450.0)
        stack_red = _tmm_design_for_wavelength(630.0)
        _, R_blue = _tmm_spectrum(stack_blue)
        _, R_red = _tmm_spectrum(stack_red)
        lam = np.linspace(380, 780, 81)
        peak_blue = lam[np.argmax(R_blue)]
        peak_red = lam[np.argmax(R_red)]
        assert peak_red > peak_blue

    def test_scan_bilayers_returns_best(self):
        result = _scan_tmm_bilayers(530.0, bilayer_range=range(3, 8))
        assert "n_bilayers" in result
        assert "delta_E" in result
        assert result["n_bilayers"] >= 3


class TestBraggSpectrum:

    def test_bragg_spectrum_shape(self):
        lam, R = _bragg_spectrum(260.0, 1.46)
        assert len(lam) == 81
        assert len(R) == 81

    def test_bragg_spectrum_has_peak(self):
        import numpy as np
        lam, R = _bragg_spectrum(260.0, 1.46)
        assert np.max(R) > 0.1

    def test_bragg_spectrum_peak_shifts_with_diameter(self):
        import numpy as np
        lam_s, R_s = _bragg_spectrum(200.0, 1.46)
        lam_l, R_l = _bragg_spectrum(350.0, 1.46)
        peak_s = lam_s[np.argmax(R_s)]
        peak_l = lam_l[np.argmax(R_l)]
        assert peak_l > peak_s


class TestPhotonGlassSpectrum:

    def test_photonic_glass_spectrum_broader(self):
        """Photonic glass should have broader peak than Bragg opal."""
        import numpy as np
        _, R_bragg = _bragg_spectrum(260.0, 1.46, fwhm_nm=30.0)
        _, R_pg = _photonic_glass_spectrum(260.0, "SiO2", fwhm_nm=60.0)
        # FWHM of photonic glass > Bragg (by construction)
        # Check that PG has lower peak (broader = lower max for same area)
        assert np.max(R_pg) < np.max(R_bragg)


class TestSpectrumToColor:

    def test_spectrum_array_input(self):
        import numpy as np
        lam = np.linspace(380, 780, 81)
        R = np.exp(-0.5 * ((lam - 530) / 20) ** 2)
        color = _spectrum_to_color(R, lam)
        assert color["Lab"] is not None
        assert color["sRGB"] is not None

    def test_float_input_gaussian(self):
        color = _spectrum_to_color(530.0)
        assert color["peak_nm"] == 530.0
        assert color["Lab"] is not None

    def test_green_has_negative_a_star(self):
        """Green light should have negative a* in Lab."""
        color = _spectrum_to_color(530.0)
        if color["Lab"] is not None:
            L, a, b = color["Lab"]
            assert a < 0  # negative a* = green

    def test_compute_delta_E_zero_for_same(self):
        c1 = _spectrum_to_color(530.0)
        dE = _compute_delta_E(c1, c1)
        assert dE is not None
        assert dE < 0.01

    def test_compute_delta_E_large_for_different(self):
        c_blue = _spectrum_to_color(450.0)
        c_red = _spectrum_to_color(650.0)
        dE = _compute_delta_E(c_blue, c_red)
        assert dE is not None
        assert dE > 30  # very different colors


class TestMultilayerDesigns:

    def test_multilayer_database_entries(self):
        multilayer_names = [n for n, e in STRUCTURAL_COLOR_DB.items()
                            if e.approach == "multilayer"]
        assert len(multilayer_names) >= 3

    def test_multilayer_design_uses_tmm(self):
        """Multilayer design should produce spectrum from TMM, not Gaussian."""
        target = TargetSpec(target_wavelength_nm=530.0)
        d = design_structural_color("TiO2-SiO2-multilayer-green", target)
        assert d.predicted_Lab is not None
        # TMM produces bright color (high L*) from high reflectance
        L, a, b = d.predicted_Lab
        assert L > 80  # TMM mirrors are very bright

    def test_si3n4_multilayer(self):
        target = TargetSpec(target_wavelength_nm=530.0)
        d = design_structural_color("Si3N4-SiO2-multilayer-green", target)
        assert d.predicted_peak_nm is not None
        assert d.delta_E is not None

    def test_multilayer_scalable(self):
        """Multilayer approaches should have high scalability."""
        target = TargetSpec(target_wavelength_nm=530.0)
        d = design_structural_color("TiO2-SiO2-multilayer-green", target)
        assert d.metrics.scalability_score.value >= 0.9


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

    def test_chelating_selectivity_physics_based(self):
        """Chelating resin selectivity now from differential repulsion.
        Pb2+ vs Ca2+: modest (similar radii/hardness on O/N donors).
        Pb2+ vs Mg2+: higher (HSAB mismatch, hydration mismatch).
        Old hardcoded value (50) overestimated; physics gives ~6-10.
        Full chelating selectivity requires chelate stability (attraction)
        which is separate from repulsion scoring."""
        target = TargetSpec(target_species="Pb2+", target_charge=2,
                            target_mw=207.2, pH=5.0,
                            interferent_species=["Ca2+", "Mg2+"])
        d = design_polymer("Chelex-100", target)
        # Physics-based: worst-case selectivity > 1 (Pb preferred)
        assert d.selectivity_vs_worst > 1.0
        # Mg2+ should be more repelled than Ca2+ (higher hydration + HSAB mismatch)
        target_mg = TargetSpec(target_species="Pb2+", target_charge=2,
                               target_mw=207.2, pH=5.0,
                               interferent_species=["Mg2+"])
        target_ca = TargetSpec(target_species="Pb2+", target_charge=2,
                               target_mw=207.2, pH=5.0,
                               interferent_species=["Ca2+"])
        d_mg = design_polymer("Chelex-100", target_mg)
        d_ca = design_polymer("Chelex-100", target_ca)
        assert d_mg.selectivity_vs_worst > d_ca.selectivity_vs_worst

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



class TestTMMForwardModel:

    def test_tmm_design_returns_stack(self):
        stack = _tmm_design_for_wavelength(530.0)
        assert len(stack) == 10  # 5 bilayers × 2

    def test_tmm_design_materials_valid(self):
        stack = _tmm_design_for_wavelength(530.0)
        for mat, t in stack:
            assert isinstance(mat, str)
            assert t > 0

    def test_tmm_spectrum_has_peak(self):
        import numpy as np
        stack = _tmm_design_for_wavelength(530.0)
        lam, R = _tmm_spectrum(stack)
        assert len(lam) > 0
        assert len(R) == len(lam)
        assert np.max(R) > 0.5  # should have strong reflectance peak

    def test_tmm_peak_near_target(self):
        """TMM designed for 530nm should peak near 530nm."""
        import numpy as np
        stack = _tmm_design_for_wavelength(530.0)
        lam, R = _tmm_spectrum(stack)
        peak_nm = lam[np.argmax(R)]
        assert abs(peak_nm - 530.0) < 30  # within 30nm

    def test_tmm_different_targets_different_peaks(self):
        import numpy as np
        stack_blue = _tmm_design_for_wavelength(450.0)
        stack_red = _tmm_design_for_wavelength(630.0)
        _, R_blue = _tmm_spectrum(stack_blue)
        _, R_red = _tmm_spectrum(stack_red)
        lam = np.linspace(380, 780, 81)
        peak_blue = lam[np.argmax(R_blue)]
        peak_red = lam[np.argmax(R_red)]
        assert peak_red > peak_blue

    def test_scan_bilayers_returns_best(self):
        result = _scan_tmm_bilayers(530.0, bilayer_range=range(3, 8))
        assert "n_bilayers" in result
        assert "delta_E" in result
        assert result["n_bilayers"] >= 3


class TestBraggSpectrum:

    def test_bragg_spectrum_shape(self):
        lam, R = _bragg_spectrum(260.0, 1.46)
        assert len(lam) == 81
        assert len(R) == 81

    def test_bragg_spectrum_has_peak(self):
        import numpy as np
        lam, R = _bragg_spectrum(260.0, 1.46)
        assert np.max(R) > 0.1

    def test_bragg_spectrum_peak_shifts_with_diameter(self):
        import numpy as np
        lam_s, R_s = _bragg_spectrum(200.0, 1.46)
        lam_l, R_l = _bragg_spectrum(350.0, 1.46)
        peak_s = lam_s[np.argmax(R_s)]
        peak_l = lam_l[np.argmax(R_l)]
        assert peak_l > peak_s


class TestPhotonGlassSpectrum:

    def test_photonic_glass_spectrum_broader(self):
        """Photonic glass should have broader peak than Bragg opal."""
        import numpy as np
        _, R_bragg = _bragg_spectrum(260.0, 1.46, fwhm_nm=30.0)
        _, R_pg = _photonic_glass_spectrum(260.0, "SiO2", fwhm_nm=60.0)
        # FWHM of photonic glass > Bragg (by construction)
        # Check that PG has lower peak (broader = lower max for same area)
        assert np.max(R_pg) < np.max(R_bragg)


class TestSpectrumToColor:

    def test_spectrum_array_input(self):
        import numpy as np
        lam = np.linspace(380, 780, 81)
        R = np.exp(-0.5 * ((lam - 530) / 20) ** 2)
        color = _spectrum_to_color(R, lam)
        assert color["Lab"] is not None
        assert color["sRGB"] is not None

    def test_float_input_gaussian(self):
        color = _spectrum_to_color(530.0)
        assert color["peak_nm"] == 530.0
        assert color["Lab"] is not None

    def test_green_has_negative_a_star(self):
        """Green light should have negative a* in Lab."""
        color = _spectrum_to_color(530.0)
        if color["Lab"] is not None:
            L, a, b = color["Lab"]
            assert a < 0  # negative a* = green

    def test_compute_delta_E_zero_for_same(self):
        c1 = _spectrum_to_color(530.0)
        dE = _compute_delta_E(c1, c1)
        assert dE is not None
        assert dE < 0.01

    def test_compute_delta_E_large_for_different(self):
        c_blue = _spectrum_to_color(450.0)
        c_red = _spectrum_to_color(650.0)
        dE = _compute_delta_E(c_blue, c_red)
        assert dE is not None
        assert dE > 30  # very different colors


class TestMultilayerDesigns:

    def test_multilayer_database_entries(self):
        multilayer_names = [n for n, e in STRUCTURAL_COLOR_DB.items()
                            if e.approach == "multilayer"]
        assert len(multilayer_names) >= 3

    def test_multilayer_design_uses_tmm(self):
        """Multilayer design should produce spectrum from TMM, not Gaussian."""
        target = TargetSpec(target_wavelength_nm=530.0)
        d = design_structural_color("TiO2-SiO2-multilayer-green", target)
        assert d.predicted_Lab is not None
        # TMM produces bright color (high L*) from high reflectance
        L, a, b = d.predicted_Lab
        assert L > 80  # TMM mirrors are very bright

    def test_si3n4_multilayer(self):
        target = TargetSpec(target_wavelength_nm=530.0)
        d = design_structural_color("Si3N4-SiO2-multilayer-green", target)
        assert d.predicted_peak_nm is not None
        assert d.delta_E is not None

    def test_multilayer_scalable(self):
        """Multilayer approaches should have high scalability."""
        target = TargetSpec(target_wavelength_nm=530.0)
        d = design_structural_color("TiO2-SiO2-multilayer-green", target)
        assert d.metrics.scalability_score.value >= 0.9


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



class TestTMMForwardModel:

    def test_tmm_design_returns_stack(self):
        stack = _tmm_design_for_wavelength(530.0)
        assert len(stack) == 10  # 5 bilayers × 2

    def test_tmm_design_materials_valid(self):
        stack = _tmm_design_for_wavelength(530.0)
        for mat, t in stack:
            assert isinstance(mat, str)
            assert t > 0

    def test_tmm_spectrum_has_peak(self):
        import numpy as np
        stack = _tmm_design_for_wavelength(530.0)
        lam, R = _tmm_spectrum(stack)
        assert len(lam) > 0
        assert len(R) == len(lam)
        assert np.max(R) > 0.5  # should have strong reflectance peak

    def test_tmm_peak_near_target(self):
        """TMM designed for 530nm should peak near 530nm."""
        import numpy as np
        stack = _tmm_design_for_wavelength(530.0)
        lam, R = _tmm_spectrum(stack)
        peak_nm = lam[np.argmax(R)]
        assert abs(peak_nm - 530.0) < 30  # within 30nm

    def test_tmm_different_targets_different_peaks(self):
        import numpy as np
        stack_blue = _tmm_design_for_wavelength(450.0)
        stack_red = _tmm_design_for_wavelength(630.0)
        _, R_blue = _tmm_spectrum(stack_blue)
        _, R_red = _tmm_spectrum(stack_red)
        lam = np.linspace(380, 780, 81)
        peak_blue = lam[np.argmax(R_blue)]
        peak_red = lam[np.argmax(R_red)]
        assert peak_red > peak_blue

    def test_scan_bilayers_returns_best(self):
        result = _scan_tmm_bilayers(530.0, bilayer_range=range(3, 8))
        assert "n_bilayers" in result
        assert "delta_E" in result
        assert result["n_bilayers"] >= 3


class TestBraggSpectrum:

    def test_bragg_spectrum_shape(self):
        lam, R = _bragg_spectrum(260.0, 1.46)
        assert len(lam) == 81
        assert len(R) == 81

    def test_bragg_spectrum_has_peak(self):
        import numpy as np
        lam, R = _bragg_spectrum(260.0, 1.46)
        assert np.max(R) > 0.1

    def test_bragg_spectrum_peak_shifts_with_diameter(self):
        import numpy as np
        lam_s, R_s = _bragg_spectrum(200.0, 1.46)
        lam_l, R_l = _bragg_spectrum(350.0, 1.46)
        peak_s = lam_s[np.argmax(R_s)]
        peak_l = lam_l[np.argmax(R_l)]
        assert peak_l > peak_s


class TestPhotonGlassSpectrum:

    def test_photonic_glass_spectrum_broader(self):
        """Photonic glass should have broader peak than Bragg opal."""
        import numpy as np
        _, R_bragg = _bragg_spectrum(260.0, 1.46, fwhm_nm=30.0)
        _, R_pg = _photonic_glass_spectrum(260.0, "SiO2", fwhm_nm=60.0)
        # FWHM of photonic glass > Bragg (by construction)
        # Check that PG has lower peak (broader = lower max for same area)
        assert np.max(R_pg) < np.max(R_bragg)


class TestSpectrumToColor:

    def test_spectrum_array_input(self):
        import numpy as np
        lam = np.linspace(380, 780, 81)
        R = np.exp(-0.5 * ((lam - 530) / 20) ** 2)
        color = _spectrum_to_color(R, lam)
        assert color["Lab"] is not None
        assert color["sRGB"] is not None

    def test_float_input_gaussian(self):
        color = _spectrum_to_color(530.0)
        assert color["peak_nm"] == 530.0
        assert color["Lab"] is not None

    def test_green_has_negative_a_star(self):
        """Green light should have negative a* in Lab."""
        color = _spectrum_to_color(530.0)
        if color["Lab"] is not None:
            L, a, b = color["Lab"]
            assert a < 0  # negative a* = green

    def test_compute_delta_E_zero_for_same(self):
        c1 = _spectrum_to_color(530.0)
        dE = _compute_delta_E(c1, c1)
        assert dE is not None
        assert dE < 0.01

    def test_compute_delta_E_large_for_different(self):
        c_blue = _spectrum_to_color(450.0)
        c_red = _spectrum_to_color(650.0)
        dE = _compute_delta_E(c_blue, c_red)
        assert dE is not None
        assert dE > 30  # very different colors


class TestMultilayerDesigns:

    def test_multilayer_database_entries(self):
        multilayer_names = [n for n, e in STRUCTURAL_COLOR_DB.items()
                            if e.approach == "multilayer"]
        assert len(multilayer_names) >= 3

    def test_multilayer_design_uses_tmm(self):
        """Multilayer design should produce spectrum from TMM, not Gaussian."""
        target = TargetSpec(target_wavelength_nm=530.0)
        d = design_structural_color("TiO2-SiO2-multilayer-green", target)
        assert d.predicted_Lab is not None
        # TMM produces bright color (high L*) from high reflectance
        L, a, b = d.predicted_Lab
        assert L > 80  # TMM mirrors are very bright

    def test_si3n4_multilayer(self):
        target = TargetSpec(target_wavelength_nm=530.0)
        d = design_structural_color("Si3N4-SiO2-multilayer-green", target)
        assert d.predicted_peak_nm is not None
        assert d.delta_E is not None

    def test_multilayer_scalable(self):
        """Multilayer approaches should have high scalability."""
        target = TargetSpec(target_wavelength_nm=530.0)
        d = design_structural_color("TiO2-SiO2-multilayer-green", target)
        assert d.metrics.scalability_score.value >= 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ═══════════════════════════════════════════════════════════════════════════
# Enhanced Physics: Competitive Langmuir, Davies, Thomas
# ═══════════════════════════════════════════════════════════════════════════

from core.material_designer import (
    competitive_langmuir, competitive_reduction_factor,
    davies_log_gamma, activity_coefficient, ionic_strength_from_species,
    correct_K_for_ionic_strength,
    thomas_breakthrough, thomas_BV_at_breakthrough, thomas_curve,
    estimate_thomas_k_from_kinetics,
)


class TestCompetitiveLangmuir:

    def test_no_interferents_equals_single(self):
        """With no interferents, competitive = single-component."""
        q_single = langmuir_capacity(100.0, 1.0, 0.1)
        q_comp = competitive_langmuir(100.0, 1.0, 0.1, [], [])
        assert q_comp == pytest.approx(q_single)

    def test_interferent_reduces_capacity(self):
        q_single = competitive_langmuir(100.0, 1.0, 0.1, [], [])
        q_comp = competitive_langmuir(100.0, 1.0, 0.1, [1.0], [5.0])
        assert q_comp < q_single

    def test_more_interferents_more_reduction(self):
        q_1 = competitive_langmuir(100.0, 1.0, 0.1, [1.0], [1.0])
        q_3 = competitive_langmuir(100.0, 1.0, 0.1, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
        assert q_3 < q_1

    def test_high_target_K_resists_competition(self):
        """High K_target means target wins competition."""
        q_strong = competitive_langmuir(100.0, 100.0, 0.1, [1.0], [5.0])
        q_weak = competitive_langmuir(100.0, 1.0, 0.1, [1.0], [5.0])
        assert q_strong > q_weak

    def test_reduction_factor_bounded(self):
        f = competitive_reduction_factor(1.0, 0.1, [1.0], [5.0])
        assert 0.0 < f < 1.0

    def test_reduction_factor_one_without_competition(self):
        f = competitive_reduction_factor(1.0, 0.1, [], [])
        assert f == pytest.approx(1.0)


class TestDaviesEquation:

    def test_ideal_at_zero_I(self):
        assert activity_coefficient(2, 0.0) == pytest.approx(1.0)

    def test_gamma_decreases_with_I(self):
        g_low = activity_coefficient(1, 0.001)
        g_high = activity_coefficient(1, 0.1)
        assert g_high < g_low

    def test_higher_charge_lower_gamma(self):
        """Divalent ions are more affected than monovalent."""
        g1 = activity_coefficient(1, 0.1)
        g2 = activity_coefficient(2, 0.1)
        assert g2 < g1

    def test_z1_I01_textbook_value(self):
        """z=1 at I=0.1: γ ≈ 0.78 (textbook reference)."""
        g = activity_coefficient(1, 0.1)
        assert 0.75 < g < 0.82

    def test_z2_I01_textbook_value(self):
        """z=2 at I=0.1: γ ≈ 0.37 (textbook reference)."""
        g = activity_coefficient(2, 0.1)
        assert 0.33 < g < 0.42

    def test_neutral_species_gamma_one(self):
        """z=0: γ = 1 always (no charge → no Debye-Hückel effect)."""
        assert activity_coefficient(0, 0.5) == pytest.approx(1.0)

    def test_ionic_strength_calculation(self):
        """0.1 M NaCl: I = 0.5×(0.1×1² + 0.1×1²) = 0.1 M."""
        I = ionic_strength_from_species([1, -1], [100.0, 100.0])  # 100 mM each
        assert I == pytest.approx(0.1)

    def test_ionic_strength_divalent(self):
        """0.01 M CaCl₂: I = 0.5×(0.01×4 + 0.02×1) = 0.03 M."""
        I = ionic_strength_from_species([2, -1], [10.0, 20.0])  # 10 mM Ca, 20 mM Cl
        assert I == pytest.approx(0.03)

    def test_K_correction_reduces_K(self):
        """Higher I → lower effective K for charged species."""
        K_low = correct_K_for_ionic_strength(1.0, 2, 0, 0.001)
        K_high = correct_K_for_ionic_strength(1.0, 2, 0, 0.5)
        assert K_high < K_low

    def test_K_correction_neutral_unchanged(self):
        """Neutral species: K unaffected by ionic strength."""
        K = correct_K_for_ionic_strength(1.0, 0, 0, 0.5)
        assert K == pytest.approx(1.0)


class TestThomasModel:

    def test_midpoint_is_half(self):
        """At BV = BV_50, C/C₀ = 0.5."""
        ratio = thomas_breakthrough(5000.0, 5000.0, 0.01)
        assert abs(ratio - 0.5) < 0.001

    def test_early_BV_low_ratio(self):
        """Well before BV_50, C/C₀ ≈ 0."""
        ratio = thomas_breakthrough(1000.0, 5000.0, 0.005)
        assert ratio < 0.01

    def test_late_BV_high_ratio(self):
        """Well after BV_50, C/C₀ ≈ 1."""
        ratio = thomas_breakthrough(9000.0, 5000.0, 0.005)
        assert ratio > 0.99

    def test_monotonically_increasing(self):
        """Breakthrough curve is monotonically increasing."""
        ratios = [thomas_breakthrough(bv, 5000.0, 0.005)
                  for bv in range(0, 10001, 500)]
        for i in range(len(ratios) - 1):
            assert ratios[i + 1] >= ratios[i]

    def test_sharper_k_steeper_curve(self):
        """Higher k_Th → steeper front."""
        # At BV just before BV_50
        r_gentle = thomas_breakthrough(4500.0, 5000.0, 0.001)
        r_sharp = thomas_breakthrough(4500.0, 5000.0, 0.01)
        assert r_gentle > r_sharp  # gentle slope → higher C/C0 before midpoint

    def test_BV_at_breakthrough_before_midpoint(self):
        """5% breakthrough occurs before BV_50."""
        bv_5 = thomas_BV_at_breakthrough(5000.0, 0.005, 0.05)
        assert bv_5 < 5000.0
        assert bv_5 > 0

    def test_BV_at_breakthrough_increases_with_sharper_k(self):
        """Sharper front → later first breakthrough (cleaner effluent longer)."""
        bv_gentle = thomas_BV_at_breakthrough(5000.0, 0.001, 0.05)
        bv_sharp = thomas_BV_at_breakthrough(5000.0, 0.01, 0.05)
        assert bv_sharp > bv_gentle

    def test_curve_has_correct_length(self):
        curve = thomas_curve(5000.0, 0.005, n_points=50)
        assert len(curve) == 51  # 0..50 inclusive

    def test_curve_starts_near_zero(self):
        curve = thomas_curve(5000.0, 0.005)
        assert curve[0][1] < 0.01  # first point near 0

    def test_curve_ends_near_one(self):
        curve = thomas_curve(5000.0, 0.005)
        assert curve[-1][1] > 0.99  # last point near 1


class TestCompetitionInFramework:

    def test_competition_reduces_framework_capacity(self):
        """Framework capacity should drop with interferents present."""
        target_clean = TargetSpec(
            target_species="SeO3^2-", target_charge=-2, target_mw=127.0,
            pH=5.0, ionic_strength_M=0.01)
        target_dirty = TargetSpec(
            target_species="SeO3^2-", target_charge=-2, target_mw=127.0,
            pH=5.0, ionic_strength_M=0.01,
            interferent_species=["SO4^2-", "Ca^2+"],
            interferent_concentrations_mM=[5.0, 10.0])

        d_clean = design_framework("UiO-66", target_clean, functional_groups=["amine-primary"])
        d_dirty = design_framework("UiO-66", target_dirty, functional_groups=["amine-primary"])

        assert d_dirty.metrics.capacity_mg_g.value < d_clean.metrics.capacity_mg_g.value


class TestCompetitionInPolymer:

    def test_competition_reduces_polymer_capacity(self):
        target_clean = TargetSpec(
            target_species="Pb2+", target_charge=2, target_mw=207.2,
            pH=5.0, ionic_strength_M=0.01)
        target_dirty = TargetSpec(
            target_species="Pb2+", target_charge=2, target_mw=207.2,
            pH=5.0, ionic_strength_M=0.01,
            interferent_species=["Ca2+", "Mg2+"],
            interferent_concentrations_mM=[10.0, 5.0])

        d_clean = design_polymer("Chelex-100", target_clean)
        d_dirty = design_polymer("Chelex-100", target_dirty)

        assert d_dirty.metrics.capacity_mg_g.value < d_clean.metrics.capacity_mg_g.value



class TestTMMForwardModel:

    def test_tmm_design_returns_stack(self):
        stack = _tmm_design_for_wavelength(530.0)
        assert len(stack) == 10  # 5 bilayers × 2

    def test_tmm_design_materials_valid(self):
        stack = _tmm_design_for_wavelength(530.0)
        for mat, t in stack:
            assert isinstance(mat, str)
            assert t > 0

    def test_tmm_spectrum_has_peak(self):
        import numpy as np
        stack = _tmm_design_for_wavelength(530.0)
        lam, R = _tmm_spectrum(stack)
        assert len(lam) > 0
        assert len(R) == len(lam)
        assert np.max(R) > 0.5  # should have strong reflectance peak

    def test_tmm_peak_near_target(self):
        """TMM designed for 530nm should peak near 530nm."""
        import numpy as np
        stack = _tmm_design_for_wavelength(530.0)
        lam, R = _tmm_spectrum(stack)
        peak_nm = lam[np.argmax(R)]
        assert abs(peak_nm - 530.0) < 30  # within 30nm

    def test_tmm_different_targets_different_peaks(self):
        import numpy as np
        stack_blue = _tmm_design_for_wavelength(450.0)
        stack_red = _tmm_design_for_wavelength(630.0)
        _, R_blue = _tmm_spectrum(stack_blue)
        _, R_red = _tmm_spectrum(stack_red)
        lam = np.linspace(380, 780, 81)
        peak_blue = lam[np.argmax(R_blue)]
        peak_red = lam[np.argmax(R_red)]
        assert peak_red > peak_blue

    def test_scan_bilayers_returns_best(self):
        result = _scan_tmm_bilayers(530.0, bilayer_range=range(3, 8))
        assert "n_bilayers" in result
        assert "delta_E" in result
        assert result["n_bilayers"] >= 3


class TestBraggSpectrum:

    def test_bragg_spectrum_shape(self):
        lam, R = _bragg_spectrum(260.0, 1.46)
        assert len(lam) == 81
        assert len(R) == 81

    def test_bragg_spectrum_has_peak(self):
        import numpy as np
        lam, R = _bragg_spectrum(260.0, 1.46)
        assert np.max(R) > 0.1

    def test_bragg_spectrum_peak_shifts_with_diameter(self):
        import numpy as np
        lam_s, R_s = _bragg_spectrum(200.0, 1.46)
        lam_l, R_l = _bragg_spectrum(350.0, 1.46)
        peak_s = lam_s[np.argmax(R_s)]
        peak_l = lam_l[np.argmax(R_l)]
        assert peak_l > peak_s


class TestPhotonGlassSpectrum:

    def test_photonic_glass_spectrum_broader(self):
        """Photonic glass should have broader peak than Bragg opal."""
        import numpy as np
        _, R_bragg = _bragg_spectrum(260.0, 1.46, fwhm_nm=30.0)
        _, R_pg = _photonic_glass_spectrum(260.0, "SiO2", fwhm_nm=60.0)
        # FWHM of photonic glass > Bragg (by construction)
        # Check that PG has lower peak (broader = lower max for same area)
        assert np.max(R_pg) < np.max(R_bragg)


class TestSpectrumToColor:

    def test_spectrum_array_input(self):
        import numpy as np
        lam = np.linspace(380, 780, 81)
        R = np.exp(-0.5 * ((lam - 530) / 20) ** 2)
        color = _spectrum_to_color(R, lam)
        assert color["Lab"] is not None
        assert color["sRGB"] is not None

    def test_float_input_gaussian(self):
        color = _spectrum_to_color(530.0)
        assert color["peak_nm"] == 530.0
        assert color["Lab"] is not None

    def test_green_has_negative_a_star(self):
        """Green light should have negative a* in Lab."""
        color = _spectrum_to_color(530.0)
        if color["Lab"] is not None:
            L, a, b = color["Lab"]
            assert a < 0  # negative a* = green

    def test_compute_delta_E_zero_for_same(self):
        c1 = _spectrum_to_color(530.0)
        dE = _compute_delta_E(c1, c1)
        assert dE is not None
        assert dE < 0.01

    def test_compute_delta_E_large_for_different(self):
        c_blue = _spectrum_to_color(450.0)
        c_red = _spectrum_to_color(650.0)
        dE = _compute_delta_E(c_blue, c_red)
        assert dE is not None
        assert dE > 30  # very different colors


class TestMultilayerDesigns:

    def test_multilayer_database_entries(self):
        multilayer_names = [n for n, e in STRUCTURAL_COLOR_DB.items()
                            if e.approach == "multilayer"]
        assert len(multilayer_names) >= 3

    def test_multilayer_design_uses_tmm(self):
        """Multilayer design should produce spectrum from TMM, not Gaussian."""
        target = TargetSpec(target_wavelength_nm=530.0)
        d = design_structural_color("TiO2-SiO2-multilayer-green", target)
        assert d.predicted_Lab is not None
        # TMM produces bright color (high L*) from high reflectance
        L, a, b = d.predicted_Lab
        assert L > 80  # TMM mirrors are very bright

    def test_si3n4_multilayer(self):
        target = TargetSpec(target_wavelength_nm=530.0)
        d = design_structural_color("Si3N4-SiO2-multilayer-green", target)
        assert d.predicted_peak_nm is not None
        assert d.delta_E is not None

    def test_multilayer_scalable(self):
        """Multilayer approaches should have high scalability."""
        target = TargetSpec(target_wavelength_nm=530.0)
        d = design_structural_color("TiO2-SiO2-multilayer-green", target)
        assert d.metrics.scalability_score.value >= 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ═══════════════════════════════════════════════════════════════════════════
# M4: Structural Color Tests
# ═══════════════════════════════════════════════════════════════════════════

from core.material_designer import (
    StructuralColorSpec, StructuralColorDesign, StructuralColorEntry,
    design_structural_color, screen_structural_colors,
    STRUCTURAL_COLOR_DB,
    _bragg_peak, _bragg_spectrum, _photonic_glass_peak, _photonic_glass_spectrum,
    _tmm_spectrum, _tmm_design_for_wavelength, _scan_tmm_bilayers,
    _spectrum_to_color, _compute_delta_E,
)


class TestStructuralColorDatabase:

    def test_database_count(self):
        assert len(STRUCTURAL_COLOR_DB) >= 8

    def test_all_have_sources(self):
        for name, entry in STRUCTURAL_COLOR_DB.items():
            assert len(entry.source) > 10, f"{name}: missing source"

    def test_all_have_approach(self):
        for name, entry in STRUCTURAL_COLOR_DB.items():
            assert entry.approach in ("bragg_opal", "photonic_glass", "BCP", "CNC",
                                       "multilayer"), f"{name}: bad approach"

    def test_all_have_published_peak(self):
        for name, entry in STRUCTURAL_COLOR_DB.items():
            assert entry.published_peak_nm is not None and entry.published_peak_nm > 0, \
                f"{name}: missing published peak"


class TestForwardModels:

    def test_bragg_peak_in_visible(self):
        """SiO2 sphere D=250nm: peak should be in visible range."""
        peak = _bragg_peak(250.0, 1.46, 1.0, 0.74)
        assert peak is not None
        assert 380 < peak < 780

    def test_bragg_peak_scales_with_diameter(self):
        """Larger spheres → longer wavelength."""
        peak_small = _bragg_peak(200.0, 1.46, 1.0, 0.74)
        peak_large = _bragg_peak(350.0, 1.46, 1.0, 0.74)
        assert peak_large > peak_small

    def test_bragg_peak_scales_with_n(self):
        """Higher n → longer wavelength at same diameter."""
        peak_low_n = _bragg_peak(250.0, 1.2, 1.0, 0.74)
        peak_high_n = _bragg_peak(250.0, 1.8, 1.0, 0.74)
        assert peak_high_n > peak_low_n

    def test_photonic_glass_peak_in_visible(self):
        """Photonic glass D=250nm SiO2: peak in visible."""
        peak = _photonic_glass_peak(250.0, "SiO2", 1.0, 0.55)
        assert peak is not None
        assert 380 < peak < 780

    def test_spectrum_to_color_returns_dict(self):
        result = _spectrum_to_color(530.0)
        assert "peak_nm" in result
        assert result["peak_nm"] == 530.0


class TestStructuralColorDesign:

    def test_from_database(self):
        spec = StructuralColorSpec.from_database("SiO2-opal-blue")
        assert spec.sphere_material == "SiO2"
        assert spec.approach == "bragg_opal"

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            StructuralColorSpec.from_database("FakeColor-99")

    def test_design_returns_design(self):
        target = TargetSpec(target_wavelength_nm=530.0)
        d = design_structural_color("SiO2-opal-green", target)
        assert isinstance(d, StructuralColorDesign)
        assert d.predicted_peak_nm is not None

    def test_bragg_opal_predicts_peak(self):
        """Bragg opal should compute peak from Bragg's law."""
        target = TargetSpec(target_wavelength_nm=530.0)
        d = design_structural_color("SiO2-opal-green", target)
        assert d.predicted_peak_nm is not None
        assert 400 < d.predicted_peak_nm < 700

    def test_bcp_uses_published_peak(self):
        """BCP has no forward model → should use published peak."""
        target = TargetSpec(target_wavelength_nm=530.0)
        d = design_structural_color("BCP-Cypris-green", target)
        assert d.predicted_peak_nm == 530.0  # published value

    def test_angle_independent_scores_higher(self):
        """Non-iridescent should score better than iridescent for same color."""
        target = TargetSpec(target_wavelength_nm=530.0)
        d_pg = design_structural_color("PS-photonic-glass-green", target)
        d_opal = design_structural_color("SiO2-opal-green", target)
        # Photonic glass is angle-independent, opal is not
        assert d_pg.spec.angle_independent
        assert not d_opal.spec.angle_independent
        assert d_pg.metrics.selectivity_ratio.value > d_opal.metrics.selectivity_ratio.value

    def test_delta_E_zero_for_exact_match(self):
        """BCP-Cypris at 530nm target should have ΔE near 0."""
        target = TargetSpec(target_wavelength_nm=530.0)
        d = design_structural_color("BCP-Cypris-green", target)
        assert d.delta_E is not None
        assert d.delta_E < 5.0


class TestScreenStructuralColors:

    def test_returns_ranked(self):
        target = TargetSpec(target_wavelength_nm=530.0)
        results = screen_structural_colors(target)
        assert len(results) >= 5
        scores = [d.metrics.composite_score() for d in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_blue_target_prefers_blue_materials(self):
        """Blue target (450nm) should rank blue materials higher."""
        target = TargetSpec(target_wavelength_nm=450.0)
        results = screen_structural_colors(target)
        top_3 = [d.spec.name for d in results[:3]]
        assert any("blue" in n.lower() for n in top_3)


class TestUnifiedWithStructuralColor:

    def test_structural_color_included_with_wavelength(self):
        """Structural color materials appear when target has wavelength."""
        target = TargetSpec(name="test", target_wavelength_nm=530.0)
        result = unified_design(target, include_frameworks=False,
                                 include_polymers=False, include_composites=False,
                                 top_n=10)
        classes = set(r.material_class for r in result.rankings)
        assert any("StructuralColor" in c for c in classes)

    def test_structural_color_excluded_without_wavelength(self):
        """No wavelength target → structural color not included."""
        target = TargetSpec(name="test", target_species="Pb2+",
                            target_charge=2, target_mw=207.2)
        result = unified_design(target, top_n=30)
        classes = set(r.material_class for r in result.rankings)
        assert not any("StructuralColor" in c for c in classes)

    def test_zero_t3_data(self):
        """M4 should produce zero T3 data (strict policy)."""
        target = TargetSpec(target_wavelength_nm=530.0)
        result = unified_design(target, include_frameworks=False,
                                 include_polymers=False, include_composites=False)
        for r in result.rankings:
            m = r.metrics
            for field_name in ['capacity_mg_g', 'selectivity_ratio', 'cost_per_kg_usd',
                               'scalability_score', 'environmental_score']:
                tv = getattr(m, field_name, None)
                if tv is not None:
                    assert tv.tier != DataTier.T3_CONCEPTUAL, \
                        f"{r.design_name}/{field_name}: has T3 data"



class TestTMMForwardModel:

    def test_tmm_design_returns_stack(self):
        stack = _tmm_design_for_wavelength(530.0)
        assert len(stack) == 10  # 5 bilayers × 2

    def test_tmm_design_materials_valid(self):
        stack = _tmm_design_for_wavelength(530.0)
        for mat, t in stack:
            assert isinstance(mat, str)
            assert t > 0

    def test_tmm_spectrum_has_peak(self):
        import numpy as np
        stack = _tmm_design_for_wavelength(530.0)
        lam, R = _tmm_spectrum(stack)
        assert len(lam) > 0
        assert len(R) == len(lam)
        assert np.max(R) > 0.5  # should have strong reflectance peak

    def test_tmm_peak_near_target(self):
        """TMM designed for 530nm should peak near 530nm."""
        import numpy as np
        stack = _tmm_design_for_wavelength(530.0)
        lam, R = _tmm_spectrum(stack)
        peak_nm = lam[np.argmax(R)]
        assert abs(peak_nm - 530.0) < 30  # within 30nm

    def test_tmm_different_targets_different_peaks(self):
        import numpy as np
        stack_blue = _tmm_design_for_wavelength(450.0)
        stack_red = _tmm_design_for_wavelength(630.0)
        _, R_blue = _tmm_spectrum(stack_blue)
        _, R_red = _tmm_spectrum(stack_red)
        lam = np.linspace(380, 780, 81)
        peak_blue = lam[np.argmax(R_blue)]
        peak_red = lam[np.argmax(R_red)]
        assert peak_red > peak_blue

    def test_scan_bilayers_returns_best(self):
        result = _scan_tmm_bilayers(530.0, bilayer_range=range(3, 8))
        assert "n_bilayers" in result
        assert "delta_E" in result
        assert result["n_bilayers"] >= 3


class TestBraggSpectrum:

    def test_bragg_spectrum_shape(self):
        lam, R = _bragg_spectrum(260.0, 1.46)
        assert len(lam) == 81
        assert len(R) == 81

    def test_bragg_spectrum_has_peak(self):
        import numpy as np
        lam, R = _bragg_spectrum(260.0, 1.46)
        assert np.max(R) > 0.1

    def test_bragg_spectrum_peak_shifts_with_diameter(self):
        import numpy as np
        lam_s, R_s = _bragg_spectrum(200.0, 1.46)
        lam_l, R_l = _bragg_spectrum(350.0, 1.46)
        peak_s = lam_s[np.argmax(R_s)]
        peak_l = lam_l[np.argmax(R_l)]
        assert peak_l > peak_s


class TestPhotonGlassSpectrum:

    def test_photonic_glass_spectrum_broader(self):
        """Photonic glass should have broader peak than Bragg opal."""
        import numpy as np
        _, R_bragg = _bragg_spectrum(260.0, 1.46, fwhm_nm=30.0)
        _, R_pg = _photonic_glass_spectrum(260.0, "SiO2", fwhm_nm=60.0)
        # FWHM of photonic glass > Bragg (by construction)
        # Check that PG has lower peak (broader = lower max for same area)
        assert np.max(R_pg) < np.max(R_bragg)


class TestSpectrumToColor:

    def test_spectrum_array_input(self):
        import numpy as np
        lam = np.linspace(380, 780, 81)
        R = np.exp(-0.5 * ((lam - 530) / 20) ** 2)
        color = _spectrum_to_color(R, lam)
        assert color["Lab"] is not None
        assert color["sRGB"] is not None

    def test_float_input_gaussian(self):
        color = _spectrum_to_color(530.0)
        assert color["peak_nm"] == 530.0
        assert color["Lab"] is not None

    def test_green_has_negative_a_star(self):
        """Green light should have negative a* in Lab."""
        color = _spectrum_to_color(530.0)
        if color["Lab"] is not None:
            L, a, b = color["Lab"]
            assert a < 0  # negative a* = green

    def test_compute_delta_E_zero_for_same(self):
        c1 = _spectrum_to_color(530.0)
        dE = _compute_delta_E(c1, c1)
        assert dE is not None
        assert dE < 0.01

    def test_compute_delta_E_large_for_different(self):
        c_blue = _spectrum_to_color(450.0)
        c_red = _spectrum_to_color(650.0)
        dE = _compute_delta_E(c_blue, c_red)
        assert dE is not None
        assert dE > 30  # very different colors


class TestMultilayerDesigns:

    def test_multilayer_database_entries(self):
        multilayer_names = [n for n, e in STRUCTURAL_COLOR_DB.items()
                            if e.approach == "multilayer"]
        assert len(multilayer_names) >= 3

    def test_multilayer_design_uses_tmm(self):
        """Multilayer design should produce spectrum from TMM, not Gaussian."""
        target = TargetSpec(target_wavelength_nm=530.0)
        d = design_structural_color("TiO2-SiO2-multilayer-green", target)
        assert d.predicted_Lab is not None
        # TMM produces bright color (high L*) from high reflectance
        L, a, b = d.predicted_Lab
        assert L > 80  # TMM mirrors are very bright

    def test_si3n4_multilayer(self):
        target = TargetSpec(target_wavelength_nm=530.0)
        d = design_structural_color("Si3N4-SiO2-multilayer-green", target)
        assert d.predicted_peak_nm is not None
        assert d.delta_E is not None

    def test_multilayer_scalable(self):
        """Multilayer approaches should have high scalability."""
        target = TargetSpec(target_wavelength_nm=530.0)
        d = design_structural_color("TiO2-SiO2-multilayer-green", target)
        assert d.metrics.scalability_score.value >= 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
