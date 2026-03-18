"""
tests/test_pka_estimator.py — Validates pH-aware pKa estimation module.
"""
import pytest
import sys, os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


class TestPkaDetection:
    """Verify pKa group identification."""

    def test_primary_amine_protonated(self):
        from core.pka_estimator import estimate_charge_at_ph
        r = estimate_charge_at_ph("NCC", 7.0)  # ethylamine, pKa~10.5
        assert r["charge_int"] == +1

    def test_carboxylic_acid_deprotonated(self):
        from core.pka_estimator import estimate_charge_at_ph
        r = estimate_charge_at_ph("CC(=O)O", 7.0)  # acetic acid, pKa~4.5
        assert r["charge_int"] == -1

    def test_phenol_neutral(self):
        from core.pka_estimator import estimate_charge_at_ph
        r = estimate_charge_at_ph("Oc1ccccc1", 7.0)  # phenol, pKa~10
        assert r["charge_int"] == 0

    def test_pyridine_neutral(self):
        from core.pka_estimator import estimate_charge_at_ph
        r = estimate_charge_at_ph("c1ccncc1", 7.0)  # pyridine, pKa~5.2
        assert r["charge_int"] == 0

    def test_diamine_plus2(self):
        from core.pka_estimator import estimate_charge_at_ph
        r = estimate_charge_at_ph("NCCCCCN", 7.0)  # cadaverine
        assert r["charge_int"] == +2

    def test_adamantanone_neutral(self):
        from core.pka_estimator import estimate_charge_at_ph
        r = estimate_charge_at_ph("O=C1CC2CC(C1)CC2", 7.0)
        assert r["charge_int"] == 0


class TestEnrichment:
    """Verify UC enrichment fires correctly."""

    def test_enriches_amine_at_cb7(self):
        from core.universal_schema import UniversalComplex
        from core.pka_estimator import enrich_uc_protonation
        uc = UniversalComplex(
            name="test", binding_mode="host_guest_inclusion",
            host_name="CB7", guest_smiles="NC12CC3CC(CC(C3)C1)C2",
        )
        changed = enrich_uc_protonation(uc)
        assert changed
        assert uc.guest_charge == 1
        assert uc.n_hbonds_formed == 3  # NH3+ → 3 donors, CB7 cap=3

    def test_skips_neutral(self):
        from core.universal_schema import UniversalComplex
        from core.pka_estimator import enrich_uc_protonation
        uc = UniversalComplex(
            name="test", binding_mode="host_guest_inclusion",
            host_name="CB7", guest_smiles="O=C1CC2CC(C1)CC2",
        )
        changed = enrich_uc_protonation(uc)
        assert not changed
        assert uc.guest_charge == 0

    def test_skips_non_hg(self):
        from core.universal_schema import UniversalComplex
        from core.pka_estimator import enrich_uc_protonation
        uc = UniversalComplex(
            name="test", binding_mode="metal_coordination",
            host_name="CB7", guest_smiles="NCC",
        )
        changed = enrich_uc_protonation(uc)
        assert not changed


class TestScorerIntegration:
    """Verify pKa enrichment fires within predict() pipeline."""

    def test_amine_gets_charged_hbond(self):
        from core.universal_schema import UniversalComplex
        from core.unified_scorer_v2 import predict
        uc = UniversalComplex(
            name="CB7:amine", binding_mode="host_guest_inclusion",
            log_Ka_exp=10.0, host_name="CB7", host_type="cucurbituril",
            is_macrocyclic=True, cavity_volume_A3=279.0,
            guest_smiles="NC12CC3CC(CC(C3)C1)C2",
            guest_volume_A3=160.0, packing_coefficient=0.575,
            guest_sasa_nonpolar_A2=180.0, guest_sasa_total_A2=257.0,
        )
        r = predict(uc)
        assert uc.guest_charge == 1, "pKa enrichment didn't fire"
        assert r.dg_hbond < -5.0, f"H-bond too weak: {r.dg_hbond}"


class TestInclusionClassifier:
    """Verify 3D inclusion depth classifier."""

    def test_adamantane_deep_cb7(self):
        from core.inclusion_classifier import classify_inclusion
        r = classify_inclusion("C12CC3CC(CC(C3)C1)C2", "CB7", 0.55)
        assert r is not None
        assert r["inclusion_depth"] > 0.85, f"depth={r['inclusion_depth']}"
        assert r["fits_portal"]

    def test_ethanol_partial_cb7(self):
        from core.inclusion_classifier import classify_inclusion
        r = classify_inclusion("CCO", "CB7", 0.20)
        assert r is not None
        assert r["inclusion_depth"] < 0.85, f"small guest should have lower radial fill"

    def test_unknown_host_returns_none(self):
        from core.inclusion_classifier import classify_inclusion
        r = classify_inclusion("CCO", "CB99", 0.5)
        assert r is None
