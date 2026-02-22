"""
tests/test_sprint15.py - Entropy decomposition and Gibbs-Helmholtz tests.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conversation.decomposer_patch import patch_targets
patch_targets()
import core.assembly_composer_patch, core.scoring_patch
import core.physics_integration, core.sprint10_integration
import core.protonation_integration, core.lfse_integration
import core.ionic_integration, core.repulsion_integration
import core.entropy_integration

from knowledge.entropy_data import ENTHALPY_FRACTIONS, DONOR_ENTROPY, WATER_RELEASE_ENTROPY
from core.entropy import decompose_thermodynamics, gibbs_helmholtz, EntropyDecomposition
from core.thermodynamics import BindingThermodynamics, R_GAS
from conversation.decomposer import decompose
from core.orchestrator import Orchestrator
from adapters.base import ToolRegistry
from adapters.dnazyme_adapter import DNAzymeAdapter
from adapters.peptide_adapter import PeptideAdapter
from adapters.aptamer_adapter import AptamerAdapter


def _build():
    r = ToolRegistry()
    r.register(DNAzymeAdapter())
    r.register(PeptideAdapter())
    r.register(AptamerAdapter())
    return r


def _make_thermo(dG_bind=-50.0, dG_desolv=10.0, dG_chelate=-18.0,
                   dG_preorg=-8.0, dG_elec=-6.0, T_k=298.15):
    dG_net = dG_bind + dG_desolv + dG_chelate + dG_preorg + dG_elec
    return BindingThermodynamics(
        dG_bind=dG_bind, dG_desolv=dG_desolv, dG_chelate=dG_chelate,
        dG_preorg=dG_preorg, dG_electrostatic=dG_elec,
        dG_net=dG_net, temperature_k=T_k,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Data tables
# ═══════════════════════════════════════════════════════════════════════════

def test_enthalpy_fractions_valid():
    """All fractions between 0 and 1."""
    for name, f in ENTHALPY_FRACTIONS.items():
        assert 0.0 <= f <= 1.0, f"{name}: f_H = {f}"
    assert ENTHALPY_FRACTIONS["dG_bind"] > 0.7
    assert ENTHALPY_FRACTIONS["dG_chelate"] < 0.3
    print("  + Enthalpy fractions valid")


def test_donor_entropy_signs():
    """N, S lose entropy; O gains (water release)."""
    assert DONOR_ENTROPY["N"] < 0
    assert DONOR_ENTROPY["S"] < 0
    assert DONOR_ENTROPY["O"] > 0
    print("  + Donor entropy signs correct")


# ═══════════════════════════════════════════════════════════════════════════
# Gibbs-Helmholtz
# ═══════════════════════════════════════════════════════════════════════════

def test_gh_identity():
    """At T_ref, ΔG = ΔH - TΔS."""
    dH, dS = -40.0, 50.0
    T = 298.15
    dG = gibbs_helmholtz(dH, dS, T, dCp=0.0, T_ref_K=T)
    expected = dH - T * dS / 1000.0
    assert abs(dG - expected) < 0.01
    print(f"  + GH identity: ΔG(T_ref) = {dG:.2f} kJ/mol")


def test_gh_entropy_driven():
    """Entropy-driven: ΔG more negative at higher T."""
    dH, dS = 10.0, 200.0
    dG_25 = gibbs_helmholtz(dH, dS, 298.15)
    dG_60 = gibbs_helmholtz(dH, dS, 333.15)
    assert dG_60 < dG_25
    print(f"  + Entropy-driven: ΔG(25°C)={dG_25:.1f}, ΔG(60°C)={dG_60:.1f}")


def test_gh_enthalpy_dominated():
    """Enthalpy-dominated: ΔG less negative at higher T."""
    dH, dS = -80.0, -100.0
    dG_25 = gibbs_helmholtz(dH, dS, 298.15)
    dG_60 = gibbs_helmholtz(dH, dS, 333.15)
    assert dG_60 > dG_25
    print(f"  + Enthalpy-dom: ΔG(25°C)={dG_25:.1f}, ΔG(60°C)={dG_60:.1f}")


def test_gh_with_dCp():
    """ΔCp adds curvature."""
    dH, dS = -40.0, 50.0
    dG_0 = gibbs_helmholtz(dH, dS, 333.15, dCp=0.0)
    dG_neg = gibbs_helmholtz(dH, dS, 333.15, dCp=-200.0)
    dG_pos = gibbs_helmholtz(dH, dS, 333.15, dCp=200.0)
    assert dG_neg != dG_0 and dG_pos != dG_0
    print(f"  + GH with ΔCp: no={dG_0:.1f}, -200={dG_neg:.1f}, +200={dG_pos:.1f}")


# ═══════════════════════════════════════════════════════════════════════════
# Decomposition
# ═══════════════════════════════════════════════════════════════════════════

def test_decomposition_sums():
    """ΔH + (-TΔS) ≈ ΔG."""
    thermo = _make_thermo()
    ed = decompose_thermodynamics(thermo, donors=["S", "S", "N"])
    recon = ed.dH_total + ed.minus_TdS_total
    assert abs(recon - ed.dG_ref) < 0.1
    print(f"  + ΔH={ed.dH_total:+.1f} + (-TΔS)={ed.minus_TdS_total:+.1f} = ΔG={ed.dG_ref:+.1f}")


def test_bind_mostly_enthalpic():
    """ΔG_bind → mostly ΔH."""
    thermo = _make_thermo(dG_bind=-50.0, dG_desolv=0, dG_chelate=0, dG_preorg=0, dG_elec=0)
    ed = decompose_thermodynamics(thermo)
    c = [x for x in ed.contributions if x.name == "dG_bind"][0]
    assert abs(c.dH) > abs(c.minus_TdS)
    print(f"  + ΔG_bind: ΔH={c.dH:+.1f}, -TΔS={c.minus_TdS:+.1f}")


def test_chelate_mostly_entropic():
    """ΔG_chelate → mostly -TΔS."""
    thermo = _make_thermo(dG_bind=0, dG_desolv=0, dG_chelate=-18.0, dG_preorg=0, dG_elec=0)
    ed = decompose_thermodynamics(thermo)
    c = [x for x in ed.contributions if x.name == "dG_chelate"][0]
    assert abs(c.minus_TdS) > abs(c.dH)
    print(f"  + ΔG_chelate: ΔH={c.dH:+.1f}, -TΔS={c.minus_TdS:+.1f}")


def test_temperature_predictions_exist():
    """Should predict ΔG at 4, 37, 60°C."""
    thermo = _make_thermo()
    ed = decompose_thermodynamics(thermo, donors=["S", "S", "N"])
    assert ed.dG_at_4C is not None
    assert ed.dG_at_37C is not None
    assert ed.dG_at_60C is not None
    print(f"  + ΔG: 4°C={ed.dG_at_4C:+.1f}, 37°C={ed.dG_at_37C:+.1f}, 60°C={ed.dG_at_60C:+.1f}")


def test_actual_temperature():
    """Target temperature produces prediction."""
    thermo = _make_thermo()
    ed = decompose_thermodynamics(thermo, donors=["S", "S", "N"], actual_temp_c=12.0)
    assert ed.dG_at_target is not None
    assert ed.target_temp_c == 12.0
    print(f"  + ΔG at AMD 12°C: {ed.dG_at_target:+.1f} kJ/mol")


def test_compensation_detection():
    """Detect enthalpy-entropy compensation."""
    thermo = _make_thermo(dG_bind=-80.0, dG_desolv=10.0, dG_chelate=0, dG_preorg=0, dG_elec=0)
    ed = decompose_thermodynamics(thermo, donors=["N", "N", "N", "N"])
    assert len(ed.enthalpy_entropy_compensation) > 0
    print(f"  + {ed.enthalpy_entropy_compensation}")


def test_temperature_sensitivity():
    """Classify temperature sensitivity."""
    thermo = _make_thermo(dG_chelate=-30.0, dG_preorg=-20.0)
    ed = decompose_thermodynamics(thermo, donors=["O", "O", "O", "O", "O", "O"])
    assert "sensitivity" in ed.temperature_sensitivity.lower()
    print(f"  + {ed.temperature_sensitivity}")


def test_cold_water_vs_hot():
    """Cold water should differ from hot for entropy-driven binding."""
    thermo = _make_thermo(dG_bind=-10.0, dG_desolv=0, dG_chelate=-30.0,
                           dG_preorg=-15.0, dG_elec=0)
    ed = decompose_thermodynamics(thermo, donors=["O", "O", "O", "O"])
    # Entropy-driven (chelate + preorg): should strengthen at higher T
    diff = ed.dG_at_60C - ed.dG_at_4C
    assert abs(diff) > 0.5, f"Should see temperature effect: Δ={diff}"
    print(f"  + ΔG(4°C)={ed.dG_at_4C:+.1f}, ΔG(60°C)={ed.dG_at_60C:+.1f} (Δ={diff:+.1f})")


# ═══════════════════════════════════════════════════════════════════════════
# E2E
# ═══════════════════════════════════════════════════════════════════════════

def test_e2e_entropy_in_reports():
    """E2E: entropy decomposition in reports."""
    registry = _build()
    prob = decompose("lead capture from acid mine drainage pH 3.5")
    prob.matrix.ionic_strength_mm = 50.0
    orch = Orchestrator(registry)
    results = orch.solve(prob)
    found = any("ENTROPY" in r.confidence_reasoning for r in results.assemblies)
    assert found, "Entropy should appear in E2E report"
    print(f"  + E2E: entropy decomposition in reports")


def test_e2e_temperature_in_reports():
    """E2E: temperature prediction for actual conditions."""
    registry = _build()
    prob = decompose("copper capture from thermal spring 60C")
    prob.matrix.temperature_c = 60.0
    prob.matrix.ionic_strength_mm = 100.0
    orch = Orchestrator(registry)
    results = orch.solve(prob)
    found = any("60" in r.confidence_reasoning and "ENTROPY" in r.confidence_reasoning
                 for r in results.assemblies)
    assert found, "Temperature prediction should appear"
    print(f"  + E2E: temperature prediction at 60°C in reports")


# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        test_enthalpy_fractions_valid,
        test_donor_entropy_signs,
        test_gh_identity,
        test_gh_entropy_driven,
        test_gh_enthalpy_dominated,
        test_gh_with_dCp,
        test_decomposition_sums,
        test_bind_mostly_enthalpic,
        test_chelate_mostly_entropic,
        test_temperature_predictions_exist,
        test_actual_temperature,
        test_compensation_detection,
        test_temperature_sensitivity,
        test_cold_water_vs_hot,
        test_e2e_entropy_in_reports,
        test_e2e_temperature_in_reports,
    ]

    print()
    print("=" * 60)
    print("  Sprint 15: Entropy Decomposition & Gibbs-Helmholtz")
    print("=" * 60)
    print()

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print()
    print(f"  Sprint 15: {passed} passed, {failed} failed")
    print()
