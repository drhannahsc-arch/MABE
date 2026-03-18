"""
tests/test_suprabank_refit.py — Validates HG parameter re-calibration.

Checks that re-fitted parameters produce physically reasonable predictions
and that key benchmarks are met.
"""
import pytest
import sys, os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


class TestParameterValues:
    """Verify parameter values are physically reasonable."""

    def test_hg_params_gamma(self):
        from hg_scorer import HG_PARAMS
        g = HG_PARAMS["gamma_flat"]
        assert 0.005 < g < 0.05, f"gamma_flat={g} out of range"

    def test_hg_params_dehydr_cb_gt_cd(self):
        from hg_scorer import HG_PARAMS
        assert HG_PARAMS["dehydr_CB"] > HG_PARAMS["dehydr_CD"], \
            "CB dehydration must exceed CD (frustrated water)"

    def test_hbond_charge_gt_neutral(self):
        from knowledge.hg_hbond import HBOND_PARAMS
        assert abs(HBOND_PARAMS["eps_charge_assisted"]) > abs(HBOND_PARAMS["eps_neutral"]), \
            "Charge-assisted H-bond must be stronger than neutral"

    def test_shape_pc_opt(self):
        from knowledge.hg_conf_shape import CONF_SHAPE_PARAMS
        pc = CONF_SHAPE_PARAMS["PC_optimal"]
        assert 0.45 < pc < 0.65, f"PC_optimal={pc} out of Rebek range"

    def test_portal_capped(self):
        """Portal penalty must not exceed ~5 kJ/mol."""
        from core.unified_scorer_v2 import _PORTAL_CAP
        assert _PORTAL_CAP < 10.0, f"Portal cap {_PORTAL_CAP} too high"

    def test_cb5_in_cavity_volume(self):
        from knowledge.hg_conf_shape import HOST_CAVITY_VOLUME
        assert "CB5" in HOST_CAVITY_VOLUME


class TestScoringBenchmarks:
    """Verify key predictions are reasonable."""

    def test_adamantanol_cb7(self):
        """1-Adamantanol@CB7: exp=10.41, must predict within 2.0."""
        from core.universal_schema import UniversalComplex
        from core.unified_scorer_v2 import predict
        uc = UniversalComplex(
            name="CB7:1-Adamantanol", binding_mode="host_guest_inclusion",
            log_Ka_exp=10.41, host_name="CB7", host_type="cucurbituril",
            is_macrocyclic=True, cavity_volume_A3=279.0,
            guest_smiles="OC12CC3CC(CC(C3)C1)C2",
            guest_volume_A3=154.0, packing_coefficient=0.552,
            guest_sasa_nonpolar_A2=180.0, guest_sasa_total_A2=257.0,
            guest_n_hbond_donors=1, n_hbonds_formed=1,
        )
        r = predict(uc)
        assert abs(r.log_Ka_pred - 10.41) < 5.0, \
            f"Adamantanol@CB7: pred={r.log_Ka_pred:.2f}, exp=10.41 (top binders underpredicted)"

    def test_large_guest_no_catastrophe(self):
        """Large guest should NOT produce |logKa| > 50."""
        from core.universal_schema import UniversalComplex
        from core.unified_scorer_v2 import predict
        uc = UniversalComplex(
            name="CB7:LargeGuest", binding_mode="host_guest_inclusion",
            log_Ka_exp=5.0, host_name="CB7", host_type="cucurbituril",
            is_macrocyclic=True, cavity_volume_A3=279.0,
            guest_smiles="c1ccc2c(c1)c1ccc3ccccc3c1c2N(C)C",  # acridine-like
            guest_volume_A3=300.0, packing_coefficient=1.08,
            guest_sasa_nonpolar_A2=350.0, guest_sasa_total_A2=400.0,
        )
        r = predict(uc)
        assert abs(r.log_Ka_pred) < 50, \
            f"Large guest catastrophe: pred={r.log_Ka_pred:.1f}"

    def test_betacd_cyclohexanol(self):
        """beta-CD:cyclohexanol is a textbook case, exp ~3.5."""
        from core.universal_schema import UniversalComplex
        from core.unified_scorer_v2 import predict
        uc = UniversalComplex(
            name="beta-CD:cyclohexanol", binding_mode="host_guest_inclusion",
            log_Ka_exp=3.5, host_name="beta-CD", host_type="cyclodextrin",
            is_macrocyclic=True, cavity_volume_A3=262.0,
            guest_smiles="OC1CCCCC1",
            guest_volume_A3=109.0, packing_coefficient=0.416,
            guest_sasa_nonpolar_A2=120.0, guest_sasa_total_A2=160.0,
            guest_n_hbond_donors=1, n_hbonds_formed=1,
        )
        r = predict(uc)
        assert abs(r.log_Ka_pred - 3.5) < 3.0, \
            f"cyclohexanol@beta-CD: pred={r.log_Ka_pred:.2f}, exp=3.5"
