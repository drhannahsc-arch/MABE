"""
tests/test_sprint13.py - Ionic strength and activity coefficient tests.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conversation.decomposer_patch import patch_targets
patch_targets()

import core.assembly_composer_patch
import core.scoring_patch
import core.physics_integration
import core.sprint10_integration
import core.protonation_integration
import core.lfse_integration
import core.ionic_integration

import math
from knowledge.ionic_data import (
    compute_activity_coefficient, debye_length_nm, davies_equation,
    debye_huckel_extended, pka_ionic_strength_correction,
    ionic_strength_from_mm, get_ion_diameter, dh_A, dh_B,
    _water_permittivity,
)
from core.ionic_strength import (
    compute_ionic_strength_correction, IonicStrengthAnalysis,
)
import core.thermodynamics as _thermo_mod
from core.assembly import RecognitionChemistry, InteriorDesign, InteriorSite
from core.problem import (
    Problem, TargetSpecies, Matrix, CompetingSpecies,
    ElectronicDescription, HydrationDescription, SizeDescription, Outcome,
)
from knowledge.structural_library import STRUCTURAL_OPTIONS
from conversation.decomposer import decompose
from core.orchestrator import Orchestrator
from adapters.base import ToolRegistry
from adapters.dnazyme_adapter import DNAzymeAdapter
from adapters.peptide_adapter import PeptideAdapter
from adapters.aptamer_adapter import AptamerAdapter


def _build():
    registry = ToolRegistry()
    registry.register(DNAzymeAdapter())
    registry.register(PeptideAdapter())
    registry.register(AptamerAdapter())
    return registry


# ═══════════════════════════════════════════════════════════════════════════
# Debye-Hückel constants
# ═══════════════════════════════════════════════════════════════════════════

def test_dh_constants_25c():
    """DH constants at 25°C match textbook values."""
    A = dh_A(25.0)
    B = dh_B(25.0)
    assert 0.50 < A < 0.52, f"A = {A}"
    assert 0.32 < B < 0.34, f"B = {B}"
    print(f"  + DH constants: A = {A:.4f}, B = {B:.4f}")


def test_dh_temperature_dependence():
    """A increases with temperature (more screening at higher T)."""
    A_10 = dh_A(10.0)
    A_25 = dh_A(25.0)
    A_50 = dh_A(50.0)
    assert A_10 < A_25 < A_50
    print(f"  + A(10°C)={A_10:.4f}, A(25°C)={A_25:.4f}, A(50°C)={A_50:.4f}")


def test_water_permittivity_range():
    """ε(H₂O) decreases with T: ~87 at 0°C, ~78 at 25°C, ~70 at 50°C."""
    eps_0 = _water_permittivity(0.0)
    eps_25 = _water_permittivity(25.0)
    eps_50 = _water_permittivity(50.0)
    assert 86 < eps_0 < 89
    assert 77 < eps_25 < 80
    assert eps_50 < eps_25
    print(f"  + ε(0°C)={eps_0:.1f}, ε(25°C)={eps_25:.1f}, ε(50°C)={eps_50:.1f}")


# ═══════════════════════════════════════════════════════════════════════════
# Activity coefficients
# ═══════════════════════════════════════════════════════════════════════════

def test_gamma_ideal_dilute():
    """γ = 1.0 at zero ionic strength."""
    g = compute_activity_coefficient("lead", 2.0, 0.0)
    assert g == 1.0
    print(f"  + γ(I=0) = {g:.3f}")


def test_gamma_decreases_with_I():
    """γ decreases as I increases."""
    g1 = compute_activity_coefficient("lead", 2.0, 1.0)
    g10 = compute_activity_coefficient("lead", 2.0, 10.0)
    g100 = compute_activity_coefficient("lead", 2.0, 100.0)
    assert g1 > g10 > g100
    assert g100 < 0.6, f"Pb²⁺ at 100 mM: γ = {g100:.3f}"
    print(f"  + Pb²⁺ γ: I=1→{g1:.3f}, I=10→{g10:.3f}, I=100→{g100:.3f}")


def test_gamma_charge_dependence():
    """Higher charge → lower γ (z² dependence)."""
    g_1 = compute_activity_coefficient("sodium", 1.0, 100.0)
    g_2 = compute_activity_coefficient("calcium", 2.0, 100.0)
    g_3 = compute_activity_coefficient("iron", 3.0, 100.0, oxidation_state=3)
    assert g_1 > g_2 > g_3
    print(f"  + γ(100 mM): Na⁺={g_1:.3f}, Ca²⁺={g_2:.3f}, Fe³⁺={g_3:.3f}")


def test_gamma_neutral():
    """Neutral species: γ = 1."""
    assert compute_activity_coefficient("complex", 0.0, 500.0) == 1.0
    print(f"  + γ(neutral) = 1.0")


def test_edh_vs_davies_low_I():
    """EDH and Davies agree at low I."""
    g_edh = debye_huckel_extended(2.0, 0.01, 5.0)
    g_dav = davies_equation(2.0, 0.01)
    assert abs(g_edh - g_dav) < 0.05
    print(f"  + I=10 mM: EDH={g_edh:.3f}, Davies={g_dav:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# Debye length
# ═══════════════════════════════════════════════════════════════════════════

def test_debye_length():
    """Debye length: ~9.6 nm at 1 mM, ~0.96 nm at 100 mM."""
    d1 = debye_length_nm(1.0)
    d100 = debye_length_nm(100.0)
    d700 = debye_length_nm(700.0)
    assert 8.0 < d1 < 11.0, f"κ⁻¹(1 mM)={d1:.1f}"
    assert 0.7 < d100 < 1.2, f"κ⁻¹(100 mM)={d100:.2f}"
    assert d700 < 0.5
    print(f"  + κ⁻¹: 1 mM→{d1:.1f} nm, 100 mM→{d100:.2f} nm, 700 mM→{d700:.2f} nm")


# ═══════════════════════════════════════════════════════════════════════════
# pKa ionic strength correction
# ═══════════════════════════════════════════════════════════════════════════

def test_pka_correction_neutral_acid():
    """Neutral acid (HA → H⁺ + A⁻): pKa increases with I."""
    pka_0 = 4.50
    pka_100 = pka_ionic_strength_correction(pka_0, 0.0, -1.0, 100.0)
    # At 100 mM, shift should be ~ +0.1 to +0.4 units for neutral acid
    assert pka_100 > pka_0, f"pKa should increase: {pka_0} → {pka_100}"
    shift = pka_100 - pka_0
    assert 0.05 < shift < 0.5, f"shift = {shift:.3f}"
    print(f"  + Carboxylate pKa: {pka_0:.2f} → {pka_100:.2f} (Δ = {shift:+.2f})")


def test_pka_correction_diprotic():
    """Diprotic acid H₂PO₄⁻(z=-1) → HPO₄²⁻(z=-2) + H⁺: pKa shifts with I."""
    # acid charge = -1, base charge = -2
    # log_γ(-1) - log_γ(-2) - log_γ(+1)
    # = -A(1)(term) - (-A(4)(term)) - (-A(1)(term))
    # = -A(term) + 4A(term) + A(term) = +4A(term) > 0 → pKa increases
    pka_0 = 7.20
    pka_100 = pka_ionic_strength_correction(pka_0, -1.0, -2.0, 100.0)
    shift = pka_100 - pka_0
    assert abs(shift) > 0.01, f"Should see shift: {pka_0} → {pka_100}"
    assert shift > 0, f"pKa should increase for this case: shift = {shift}"
    print(f"  + H₂PO₄⁻ pKa: {pka_0:.2f} → {pka_100:.2f} (Δ = {shift:+.2f})")


# ═══════════════════════════════════════════════════════════════════════════
# Composite ionic strength analysis
# ═══════════════════════════════════════════════════════════════════════════

def _make_pb_problem(ionic_strength_mm=50.0):
    target = TargetSpecies(
        identity="lead", formula="Pb2+", charge=2.0,
        geometry="hemidirected",
        electronic=ElectronicDescription(
            hardness_softness="borderline", electronegativity=2.33,
        ),
        hydration=HydrationDescription(
            dehydration_energy_kj_mol=1481.0, coordination_number_water=6,
        ),
        size=SizeDescription(ionic_radius_angstrom=1.19),
    )
    matrix = Matrix(
        description="AMD", ph=3.5, temperature_c=12.0,
        ionic_strength_mm=ionic_strength_mm,
        competing_species=[
            CompetingSpecies(identity="calcium", formula="Ca2+", charge=2.0, concentration_mm=5.0),
            CompetingSpecies(identity="sodium", formula="Na+", charge=1.0, concentration_mm=2.0),
            CompetingSpecies(identity="iron", formula="Fe3+", charge=3.0, concentration_mm=1.0),
        ],
    )
    outcome = Outcome(description="capture_release")
    return Problem(target=target, matrix=matrix, desired_outcome=outcome)


def _make_recognition(donors=None, donor_type=None):
    donors = donors or ["S", "S", "N"]
    donor_type = donor_type or "soft"
    return RecognitionChemistry(
        name="dithiol_amine", type="chelator",
        donor_atoms=donors, donor_type=donor_type,
        structure="dithiol_amine",
    )


def test_ionic_correction_amd():
    """AMD at 50 mM: should see significant γ correction."""
    prob = _make_pb_problem(50.0)
    rec = _make_recognition()
    ionic = compute_ionic_strength_correction(rec, prob.target, prob)
    assert ionic.gamma_target < 0.7, f"γ = {ionic.gamma_target}"
    assert ionic.debye_length_nm < 2.0
    assert abs(ionic.dG_activity) > 0.5, f"ΔG_act = {ionic.dG_activity}"
    print(f"  + AMD (50 mM): γ={ionic.gamma_target:.3f}, "
          f"κ⁻¹={ionic.debye_length_nm:.1f} nm, "
          f"ΔG_act={ionic.dG_activity:+.1f} kJ/mol")


def test_ionic_correction_seawater():
    """Seawater at 700 mM: extreme screening."""
    prob = _make_pb_problem(700.0)
    rec = _make_recognition()
    ionic = compute_ionic_strength_correction(rec, prob.target, prob)
    assert ionic.gamma_target < 0.4
    assert ionic.debye_length_nm < 0.5
    print(f"  + Seawater (700 mM): γ={ionic.gamma_target:.3f}, "
          f"κ⁻¹={ionic.debye_length_nm:.2f} nm")


def test_ionic_correction_dilute():
    """Clean river at 1 mM: minimal correction."""
    prob = _make_pb_problem(1.0)
    rec = _make_recognition()
    ionic = compute_ionic_strength_correction(rec, prob.target, prob)
    assert ionic.gamma_target > 0.8
    assert ionic.debye_length_nm > 5.0
    print(f"  + Dilute (1 mM): γ={ionic.gamma_target:.3f}, "
          f"κ⁻¹={ionic.debye_length_nm:.1f} nm")


def test_selectivity_differential():
    """Na⁺ (z=1) should have higher γ than Pb²⁺ (z=2) at same I."""
    prob = _make_pb_problem(100.0)
    rec = _make_recognition()
    ionic = compute_ionic_strength_correction(rec, prob.target, prob)
    g_pb = ionic.gamma_target
    g_na = compute_activity_coefficient("sodium", 1.0, 100.0)
    assert g_na > g_pb, f"Na⁺ γ={g_na:.3f} should > Pb²⁺ γ={g_pb:.3f}"
    print(f"  + Selectivity differential: γ(Pb²⁺)={g_pb:.3f} vs γ(Na⁺)={g_na:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# Thermodynamics integration
# ═══════════════════════════════════════════════════════════════════════════

def test_thermo_with_ionic_strength():
    """ΔG_net should change when ionic strength is present."""
    prob_dilute = _make_pb_problem(0.001)
    prob_amd = _make_pb_problem(50.0)
    rec = _make_recognition()
    struct = [s for s in STRUCTURAL_OPTIONS if s.type == "mesoporous_silica"][0]
    interior = InteriorDesign(
        design_level="primary",
        sites=[InteriorSite(recognition=rec)],
        avidity_factor=1.0,
    )
    thermo_dilute = _thermo_mod.compute_thermodynamics(rec, struct, interior, prob_dilute)
    thermo_amd = _thermo_mod.compute_thermodynamics(rec, struct, interior, prob_amd)
    diff = abs(thermo_amd.dG_net - thermo_dilute.dG_net)
    assert diff > 0.1, f"Ionic strength should change ΔG: dilute={thermo_dilute.dG_net:.1f}, AMD={thermo_amd.dG_net:.1f}"
    print(f"  + ΔG: dilute={thermo_dilute.dG_net:+.1f}, AMD(50 mM)={thermo_amd.dG_net:+.1f} kJ/mol (Δ={diff:.1f})")


# ═══════════════════════════════════════════════════════════════════════════
# E2E
# ═══════════════════════════════════════════════════════════════════════════

def test_e2e_ionic_in_reports():
    """E2E: ionic strength data appears in confidence_reasoning."""
    registry = _build()
    prob = decompose("lead capture from acid mine drainage pH 3.5 ionic strength 50 mM")
    prob.matrix.ionic_strength_mm = 50.0
    orch = Orchestrator(registry)
    results = orch.solve(prob)
    found_ionic = False
    for r in results.assemblies:
        if "IONIC STRENGTH" in r.confidence_reasoning or "ionic" in r.confidence_reasoning.lower():
            found_ionic = True
            break
    assert found_ionic, "Ionic strength should appear in E2E report"
    print(f"  + E2E: ionic strength data in reports ({len(results.assemblies)} assemblies)")


def test_e2e_seawater_warning():
    """E2E: high ionic strength should trigger warning."""
    registry = _build()
    prob = decompose("lead capture from seawater")
    prob.matrix.ionic_strength_mm = 700.0
    orch = Orchestrator(registry)
    results = orch.solve(prob)
    found_warning = False
    for r in results.assemblies:
        for fm in r.failure_modes:
            if "ionic strength" in fm.lower() or "activity coefficient" in fm.lower() or "davies" in fm.lower():
                found_warning = True
    assert found_warning, "Seawater should trigger high I warning"
    print(f"  + E2E: seawater warning found")


# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        test_dh_constants_25c,
        test_dh_temperature_dependence,
        test_water_permittivity_range,
        test_gamma_ideal_dilute,
        test_gamma_decreases_with_I,
        test_gamma_charge_dependence,
        test_gamma_neutral,
        test_edh_vs_davies_low_I,
        test_debye_length,
        test_pka_correction_neutral_acid,
        test_pka_correction_diprotic,
        test_ionic_correction_amd,
        test_ionic_correction_seawater,
        test_ionic_correction_dilute,
        test_selectivity_differential,
        test_thermo_with_ionic_strength,
        test_e2e_ionic_in_reports,
        test_e2e_seawater_warning,
    ]

    print()
    print("=" * 60)
    print("  Sprint 13: Ionic Strength & Activity Coefficients")
    print("=" * 60)
    print()

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print(f"  Sprint 13: {passed} passed, {failed} failed")
    print()
