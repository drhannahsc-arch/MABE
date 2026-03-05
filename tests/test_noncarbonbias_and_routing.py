"""
tests/test_noncarbonbias_and_routing.py

Tests for:
1. noncarbonbias_donors.py — full periodic table donor registry
2. scorer_frozen.py — new donor types, new metals
3. universal_predictor.py — routing wall removal, cross-modal terms
4. physics_realization_bridge.py — expanded _MATERIAL_DONOR_DEFAULTS
"""
import math
import pytest
import sys
import os

# Ensure MABE root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ──────────────────────────────────────────────────────────────────────────
# 1. noncarbonbias_donors.py
# ──────────────────────────────────────────────────────────────────────────

class TestNonCarbonBiasDonors:

    def test_import(self):
        from core.noncarbonbias_donors import (
            DONOR_INTRINSIC, DONOR_SOFTNESS, DONOR_STERIC_RADIUS,
            DONOR_H_COORD, DONOR_POLARIZABILITY,
            ALL_KNOWN_DONORS, get_donor_properties, classify_donor
        )

    def test_chalcogenide_coverage(self):
        from core.noncarbonbias_donors import DONOR_INTRINSIC, DONOR_SOFTNESS
        for subtype in [
            "Se_selenolate", "Se_selenoether", "Se_selenourea",
            "Te_tellurolate", "Te_telluroether"
        ]:
            assert subtype in DONOR_INTRINSIC, f"Missing intrinsic: {subtype}"
            assert subtype in DONOR_SOFTNESS, f"Missing softness: {subtype}"

    def test_softness_ordering_chalcogenides(self):
        """Softness must follow O < S < Se < Te (physics-first ordering)."""
        from core.noncarbonbias_donors import DONOR_SOFTNESS
        assert DONOR_SOFTNESS["O_carboxylate"] < DONOR_SOFTNESS["S_thiolate"]
        assert DONOR_SOFTNESS["S_thiolate"] < DONOR_SOFTNESS["Se_selenolate"]
        assert DONOR_SOFTNESS["Se_selenolate"] < DONOR_SOFTNESS["Te_tellurolate"]

    def test_softness_ordering_pnictogens(self):
        """P ~ As ~ Sb in softness; all softer than N."""
        from core.noncarbonbias_donors import DONOR_SOFTNESS
        assert DONOR_SOFTNESS["N_amine"] < DONOR_SOFTNESS["P_phosphine"]
        assert DONOR_SOFTNESS["P_phosphine"] < DONOR_SOFTNESS["As_arsine"]
        # Sb slightly softer than As (larger atom, more diffuse)
        assert DONOR_SOFTNESS["As_arsine"] >= DONOR_SOFTNESS["Sb_stibine"] - 0.1

    def test_fluoride_is_hard(self):
        from core.noncarbonbias_donors import DONOR_SOFTNESS
        assert DONOR_SOFTNESS["F_fluoride"] < 0.05

    def test_carbonyl_soft(self):
        from core.noncarbonbias_donors import DONOR_SOFTNESS
        assert DONOR_SOFTNESS["C_carbonyl"] > 0.80

    def test_get_donor_properties_complete(self):
        from core.noncarbonbias_donors import get_donor_properties
        props = get_donor_properties("Se_selenolate")
        assert set(props.keys()) == {
            "intrinsic", "softness", "steric_radius", "h_coord", "polarizability"
        }
        assert props["intrinsic"] < 0  # Negative (favourable)
        assert 0 < props["softness"] <= 1

    def test_classify_donor(self):
        from core.noncarbonbias_donors import classify_donor
        assert classify_donor("Se_selenolate") == "Se"
        assert classify_donor("Te_tellurolate") == "Te"
        assert classify_donor("As_arsine") == "As"
        assert classify_donor("F_fluoride") == "F"
        assert classify_donor("C_carbonyl") == "C"
        assert classify_donor("O_carboxylate") == "O"

    def test_intrinsic_energy_ordering(self):
        """Selenolate should bind more strongly than thiolate for soft metals."""
        from core.noncarbonbias_donors import DONOR_INTRINSIC
        # More negative = more favourable
        assert DONOR_INTRINSIC["Se_selenolate"] < DONOR_INTRINSIC["S_thiolate"]
        assert DONOR_INTRINSIC["Te_tellurolate"] < DONOR_INTRINSIC["Se_selenolate"]


# ──────────────────────────────────────────────────────────────────────────
# 2. scorer_frozen.py — new donor types and metals
# ──────────────────────────────────────────────────────────────────────────

class TestScorerFrozenExpansion:

    def test_new_metals_registered(self):
        from core.scorer_frozen import METAL_DB
        new_metals = [
            "Ru2+", "Ru3+", "Os2+", "Rh3+", "Ir3+",
            "Mo3+", "W4+", "Re3+", "Ge2+", "Zr4+", "Hf4+",
            "Ti4+", "Be2+",
        ]
        for m in new_metals:
            assert m in METAL_DB, f"Metal {m} not in METAL_DB"

    def test_new_metals_properties_sensible(self):
        from core.scorer_frozen import METAL_DB
        # Ru2+ should be softer than Ca2+
        assert METAL_DB["Ru2+"].hsab_softness > METAL_DB["Ca2+"].hsab_softness
        # Zr4+ should be hard
        assert METAL_DB["Zr4+"].hsab_softness < 0.20
        # Be2+ should be hard and small
        assert METAL_DB["Be2+"].hsab_softness < 0.15
        assert METAL_DB["Be2+"].ionic_radius_pm < 40

    def test_predict_with_selenolate(self):
        """Se_selenolate should produce valid (non-zero) log K for Hg2+."""
        from core.scorer_frozen import predict_log_k
        lk = predict_log_k("Hg2+", ["Se_selenolate", "Se_selenolate"], pH=7.0)
        assert lk > 0, f"Hg2+/Se_selenolate should be positive log K, got {lk}"

    def test_selenolate_stronger_than_thiolate_for_hg(self):
        """Se_selenolate × 2 must give higher log K than S_thiolate × 2 for Hg2+."""
        from core.scorer_frozen import predict_log_k
        lk_se = predict_log_k("Hg2+", ["Se_selenolate", "Se_selenolate"], pH=7.0)
        lk_s = predict_log_k("Hg2+", ["S_thiolate", "S_thiolate"], pH=7.0)
        assert lk_se > lk_s, (
            f"Se_selenolate ({lk_se:.2f}) should beat S_thiolate ({lk_s:.2f}) "
            f"for Hg2+ (soft metal + soft donor = HSAB match)"
        )

    def test_fluoride_selective_for_hard_metals(self):
        """F_fluoride should give higher log K for Al3+ than for Hg2+."""
        from core.scorer_frozen import predict_log_k
        lk_al = predict_log_k("Al3+", ["F_fluoride"] * 4, pH=7.0)
        lk_hg = predict_log_k("Hg2+", ["F_fluoride"] * 4, pH=7.0)
        assert lk_al > lk_hg, (
            f"F_fluoride should prefer hard Al3+ ({lk_al:.2f}) "
            f"over soft Hg2+ ({lk_hg:.2f})"
        )

    def test_cyanide_for_soft_metals(self):
        """C_cyanide (C-end) should bind Fe2+ more strongly than Ca2+."""
        from core.scorer_frozen import predict_log_k
        lk_fe = predict_log_k("Fe2+", ["C_cyanide"] * 6, pH=7.0)
        lk_ca = predict_log_k("Ca2+", ["C_cyanide"] * 6, pH=7.0)
        assert lk_fe > lk_ca, (
            f"C_cyanide should prefer Fe2+ ({lk_fe:.2f}) over Ca2+ ({lk_ca:.2f})"
        )

    def test_ru_and_rh_predict_without_error(self):
        """New 4d metals should score without exceptions."""
        from core.scorer_frozen import predict_log_k
        for metal in ["Ru2+", "Ru3+", "Rh3+", "Ir3+", "Os2+"]:
            lk = predict_log_k(metal, ["N_amine"] * 4 + ["O_carboxylate"] * 2, pH=7.0)
            assert isinstance(lk, float), f"{metal} predict returned non-float"
            assert not math.isnan(lk), f"{metal} predict returned NaN"

    def test_phosphine_arsine_selectivity(self):
        """P_phosphine and As_arsine should both score higher than N_amine for Pd2+."""
        from core.scorer_frozen import predict_log_k
        lk_p = predict_log_k("Pd2+", ["P_phosphine"] * 4, pH=7.0)
        lk_as = predict_log_k("Pd2+", ["As_arsine"] * 4, pH=7.0)
        lk_n = predict_log_k("Pd2+", ["N_amine"] * 4, pH=7.0)
        assert lk_p > lk_n, f"P_phosphine ({lk_p:.1f}) should beat N_amine ({lk_n:.1f}) for Pd2+"
        assert lk_as > lk_n, f"As_arsine ({lk_as:.1f}) should beat N_amine ({lk_n:.1f}) for Pd2+"


# ──────────────────────────────────────────────────────────────────────────
# 3. universal_predictor.py — routing wall removal
# ──────────────────────────────────────────────────────────────────────────

class TestRoutingWallRemoved:

    def _make_metal_hg_uc(self):
        """Metal entry WITH cavity (cross-modal)."""
        from core.universal_schema import UniversalComplex
        return UniversalComplex(
            name="Hg2+_in_betaCD",
            binding_mode="cross_modal",
            log_Ka_exp=3.5,
            metal_formula="Hg2+",
            metal_charge=2,
            donor_subtypes=["S_thiolate", "S_thiolate"],
            host_name="beta-cyclodextrin",
            host_type="cyclodextrin",
            is_macrocyclic=True,
            cavity_volume_A3=262.0,
            cavity_radius_nm=0.39,
            n_hbond_acceptors_host=7,
            guest_sasa_nonpolar_A2=45.0,
            guest_sasa_polar_A2=10.0,
            guest_sasa_total_A2=55.0,
            guest_volume_A3=22.0,
        )

    def _make_pure_metal_uc(self):
        """Pure metal coordination (no cavity)."""
        from core.universal_schema import UniversalComplex
        return UniversalComplex(
            name="Cu2+_EDTA_test",
            binding_mode="metal_coordination",
            log_Ka_exp=18.8,
            metal_formula="Cu2+",
            metal_charge=2,
            donor_subtypes=["N_amine", "N_amine", "O_carboxylate", "O_carboxylate"],
            chelate_rings=3,
        )

    def test_metal_uc_gets_hydrophobic_term_when_sasa_present(self):
        """After routing wall removal, metal entry with SASA data gets hydrophobic term."""
        from core.universal_predictor import predict
        uc = self._make_metal_hg_uc()
        result = predict(uc)
        # Hydrophobic term should be non-zero (SASA > 0 + cavity present)
        assert result.dg_hydrophobic < 0, (
            f"Cross-modal entry should have negative dg_hydrophobic, "
            f"got {result.dg_hydrophobic}"
        )

    def test_pure_metal_without_cavity_no_hydrophobic(self):
        """Pure metal coordination without SASA → hydrophobic term stays zero."""
        from core.universal_predictor import predict
        uc = self._make_pure_metal_uc()
        result = predict(uc)
        assert result.dg_hydrophobic == 0.0, (
            f"Pure metal with no SASA data should have zero hydrophobic, "
            f"got {result.dg_hydrophobic}"
        )

    def test_metal_terms_still_fire_in_cross_modal(self):
        """Metal binding terms must still activate in cross-modal entries."""
        from core.universal_predictor import predict
        uc = self._make_metal_hg_uc()
        result = predict(uc)
        # Metal terms produce dg_bind
        assert result.dg_bind != 0.0, "Metal terms should fire in cross_modal mode"

    def test_prediction_finite_for_all_modes(self):
        """No NaN or Inf in any prediction result field."""
        from core.universal_predictor import predict
        for uc in [self._make_metal_hg_uc(), self._make_pure_metal_uc()]:
            result = predict(uc)
            assert not math.isnan(result.dg_pred_kj), f"NaN in {uc.name}"
            assert not math.isinf(result.dg_pred_kj), f"Inf in {uc.name}"


# ──────────────────────────────────────────────────────────────────────────
# 4. Cross-modal terms
# ──────────────────────────────────────────────────────────────────────────

class TestCrossModalTerms:

    def _make_ba_cb7_uc(self):
        """Ba2+@CB[7] — canonical cross-modal validation case."""
        from core.universal_schema import UniversalComplex
        return UniversalComplex(
            name="Ba2+_CB7",
            binding_mode="cross_modal",
            log_Ka_exp=3.8,   # Barrow 2015
            metal_formula="Ba2+",
            metal_charge=2,
            donor_subtypes=["O_carbonyl"] * 4,
            host_name="CB7",
            host_type="cucurbituril",
            is_macrocyclic=True,
            cavity_volume_A3=279.0,
            cavity_radius_nm=0.37,
            n_hbond_acceptors_host=14,  # CB[7] has 14 portal C=O groups
            guest_volume_A3=18.0,       # Ba2+ effective volume
        )

    def test_ion_dipole_fires_for_metal_in_macrocycle(self):
        from core.universal_predictor import predict, _compute_cross_modal_ion_dipole, PhysicsParameters
        from core.universal_schema import UniversalComplex
        uc = self._make_ba_cb7_uc()
        params = PhysicsParameters()
        from core.universal_predictor import PredictionResult
        result = PredictionResult(
            name=uc.name, binding_mode=uc.binding_mode,
            log_Ka_exp=uc.log_Ka_exp, log_Ka_pred=0.0, dg_pred_kj=0.0, error=0.0
        )
        result = _compute_cross_modal_ion_dipole(uc, params, result)
        assert result.dg_ion_dipole < 0, (
            f"Ion-dipole should be negative for Ba2+@CB7, got {result.dg_ion_dipole}"
        )

    def test_ion_dipole_zero_for_pure_hg_uc(self):
        """No macrocyclic host → ion-dipole term must be zero."""
        from core.universal_predictor import _compute_cross_modal_ion_dipole, PhysicsParameters, PredictionResult
        from core.universal_schema import UniversalComplex
        uc = UniversalComplex(
            name="Hg2+_simple",
            binding_mode="metal_coordination",
            log_Ka_exp=10.0,
            metal_formula="Hg2+", metal_charge=2,
            donor_subtypes=["S_thiolate", "S_thiolate"],
        )
        params = PhysicsParameters()
        result = PredictionResult(
            name=uc.name, binding_mode=uc.binding_mode,
            log_Ka_exp=uc.log_Ka_exp, log_Ka_pred=0.0, dg_pred_kj=0.0, error=0.0
        )
        result = _compute_cross_modal_ion_dipole(uc, params, result)
        assert result.dg_ion_dipole == 0.0

    def test_portal_size_match_fires_for_cb7(self):
        from core.universal_predictor import _compute_portal_size_match, PhysicsParameters, PredictionResult
        uc = self._make_ba_cb7_uc()
        params = PhysicsParameters()
        result = PredictionResult(
            name=uc.name, binding_mode=uc.binding_mode,
            log_Ka_exp=uc.log_Ka_exp, log_Ka_pred=0.0, dg_pred_kj=0.0, error=0.0
        )
        result = _compute_portal_size_match(uc, params, result)
        assert result.dg_portal_size <= 0, (
            f"Portal size match should be <= 0, got {result.dg_portal_size}"
        )

    def test_full_predict_ba_cb7_sign_correct(self):
        """Full prediction for Ba2+@CB7 should give positive log Ka (binding occurs)."""
        from core.universal_predictor import predict
        uc = self._make_ba_cb7_uc()
        result = predict(uc)
        assert result.log_Ka_pred > 0, (
            f"Ba2+@CB7 should have positive predicted log Ka, got {result.log_Ka_pred}"
        )


# ──────────────────────────────────────────────────────────────────────────
# 5. physics_realization_bridge.py — expanded _MATERIAL_DONOR_DEFAULTS
# ──────────────────────────────────────────────────────────────────────────

class TestMaterialDonorDefaults:

    def test_chalcogenide_mof_keys(self):
        from core.physics_realization_bridge import _MATERIAL_DONOR_DEFAULTS
        assert "chalcogenide_mof" in _MATERIAL_DONOR_DEFAULTS
        d = _MATERIAL_DONOR_DEFAULTS["chalcogenide_mof"]
        assert "Se" in d
        assert "Te" in d
        assert d["Se"] == "Se_selenolate"

    def test_fluoride_host_keys(self):
        from core.physics_realization_bridge import _MATERIAL_DONOR_DEFAULTS
        assert "fluoride_host" in _MATERIAL_DONOR_DEFAULTS
        d = _MATERIAL_DONOR_DEFAULTS["fluoride_host"]
        assert "F" in d

    def test_resolve_se_donor(self):
        from core.physics_realization_bridge import _resolve_donor_subtypes
        subtypes = _resolve_donor_subtypes(["Se", "Se", "N"], "chalcogenide_mof")
        assert subtypes[0] == "Se_selenolate"
        assert subtypes[2] == "N_aromatic"

    def test_resolve_fallback_te(self):
        """Te not in any specific material system → fallback."""
        from core.physics_realization_bridge import _resolve_donor_subtypes
        subtypes = _resolve_donor_subtypes(["Te"], "crown_ether")
        # crown_ether doesn't have Te → fallback
        assert subtypes[0] == "Te_telluroether"

    def test_resolve_already_subtype(self):
        """If input already has '_', it should pass through unchanged."""
        from core.physics_realization_bridge import _resolve_donor_subtypes
        subtypes = _resolve_donor_subtypes(["Se_selenolate", "N_amine"], "chalcogenide_mof")
        assert subtypes[0] == "Se_selenolate"
        assert subtypes[1] == "N_amine"

    def test_phosphine_resin_keys(self):
        from core.physics_realization_bridge import _MATERIAL_DONOR_DEFAULTS
        assert "phosphine_resin" in _MATERIAL_DONOR_DEFAULTS
        d = _MATERIAL_DONOR_DEFAULTS["phosphine_resin"]
        assert d["P"] == "P_phosphine"

    def test_existing_material_systems_preserved(self):
        """Original material systems must still be present and unchanged."""
        from core.physics_realization_bridge import _MATERIAL_DONOR_DEFAULTS
        for system in [
            "planar_coordination_ring", "cyclic_encapsulant",
            "periodic_lattice_node", "crown_ether", "cyclodextrin"
        ]:
            assert system in _MATERIAL_DONOR_DEFAULTS, f"Missing: {system}"
