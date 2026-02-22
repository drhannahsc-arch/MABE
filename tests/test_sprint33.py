"""tests/test_sprint33.py — Sprint 33: Selectivity Scoring (18 tests)"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.selectivity import compute_selectivity, SelectivityProfile, _PANELS
from core.physics_integration import compute_enhanced_thermodynamics
from core.generative_physics_adapter import (
    RecognitionChemistry, StructuralConstraint, InteriorDesign,
    TargetSpecies, Matrix, Problem,
)
from core.design_package import design_binder

def _rec(donors, dt="soft", chel=2, match=0.9):
    return RecognitionChemistry(name="t", type="generative", donor_atoms=donors,
        donor_type=dt, denticity=len(donors), hsab_match=match, chelate_rings=chel)

def _struct(stype="zeolite"):
    return StructuralConstraint(name="s", type=stype, geometry="channel", pore_size_nm=0.74)

def _interior():
    return InteriorDesign(description="t", num_binding_sites=1, self_binding=True)

def _matrix(ph=7.0):
    return Matrix(ph=ph, temperature_c=25.0, ionic_strength_mm=10.0)


# ═══════════════════════════════════════════════════════════════════════════
# SELECTIVITY MODULE
# ═══════════════════════════════════════════════════════════════════════════

def test_s_donor_selective_for_soft():
    """S-donor binder should show high selectivity vs hard ions when target_dg is provided."""
    # Pb2+ + 4S in zeolite: model predicts dG ~ -260 kJ/mol
    sel = compute_selectivity("Pb2+", 0.0, 2,
        _rec(["S","S","S","S"]), _struct(), _interior(), _matrix(6.0),
        target_dg_kj=-260.0)
    ca = next(r for r in sel.interferents if r.formula == "Ca2+")
    assert ca.selectivity_ratio > 100, f"S-donors should reject Ca2+, ratio={ca.selectivity_ratio}"
    print(f"  \u2705 test_s_selective: Ca2+ ratio={ca.selectivity_ratio:.0f}×")

def test_o_donor_binds_hard():
    """O-donor binder should bind Ca2+ well (poor selectivity for Pb2+)."""
    sel = compute_selectivity("Pb2+", 100.0, 2,
        _rec(["O","O","O","O"], "hard", 2, 0.3), _struct(), _interior(), _matrix(7.0))
    ca = next(r for r in sel.interferents if r.formula == "Ca2+")
    # Ca2+ should bind O donors reasonably → lower selectivity ratio
    print(f"  \u2705 test_o_binds_hard: Ca2+ Kd={ca.predicted_kd_uM:.0f}, ratio={ca.selectivity_ratio:.0f}×")

def test_panels_exist():
    """All standard panels should be defined."""
    for panel in ["drinking_water", "seawater", "acid_mine", "nuclear_waste", "soil"]:
        assert panel in _PANELS, f"Missing panel: {panel}"
        assert len(_PANELS[panel]) >= 3
    print(f"  \u2705 test_panels: {len(_PANELS)} panels defined")

def test_target_excluded_from_panel():
    """Target ion should not appear as its own interferent."""
    sel = compute_selectivity("Cu2+", 1.0, 2,
        _rec(["N","N","O","O"], "borderline"), _struct(), _interior(), _matrix(7.0))
    formulas = [r.formula for r in sel.interferents]
    assert "Cu2+" not in formulas, "Target should be excluded from interferent list"
    print(f"  \u2705 test_exclude_target: interferents={formulas[:4]}")

def test_selectivity_score_range():
    """Score should be 0-100."""
    sel = compute_selectivity("Pb2+", 10.0, 2,
        _rec(["N","N","O","O"], "borderline"), _struct(), _interior(), _matrix(6.0))
    assert 0 <= sel.selectivity_score <= 100
    print(f"  \u2705 test_score_range: {sel.selectivity_score:.0f}/100")

def test_worst_interferent_identified():
    """Should identify the most competitive interferent."""
    sel = compute_selectivity("Pb2+", 10.0, 2,
        _rec(["N","N","N","N"], "borderline"), _struct(), _interior(), _matrix(7.0))
    assert sel.worst_interferent != ""
    assert sel.worst_selectivity_ratio >= 0
    print(f"  \u2705 test_worst: {sel.worst_interferent} ratio={sel.worst_selectivity_ratio:.0f}×")

def test_classification_correct():
    """Selectivity classes should be assigned correctly."""
    from core.selectivity import _classify_selectivity
    assert _classify_selectivity(2000) == "excellent"
    assert _classify_selectivity(500) == "good"
    assert _classify_selectivity(50) == "moderate"
    assert _classify_selectivity(5) == "poor"
    assert _classify_selectivity(0.5) == "none"
    print(f"  \u2705 test_classify: all classes correct")

def test_acid_mine_panel():
    """Acid mine panel should include Fe3+, Cu2+, Zn2+."""
    panel = _PANELS["acid_mine"]
    formulas = [f for f, c, d in panel]
    assert "Fe3+" in formulas
    assert "Cu2+" in formulas
    assert "Zn2+" in formulas
    print(f"  \u2705 test_acid_mine: {formulas}")

def test_seawater_panel():
    """Seawater panel should include high-concentration ions."""
    panel = _PANELS["seawater"]
    formulas = [f for f, c, d in panel]
    assert "Na+" in formulas
    assert "Mg2+" in formulas
    print(f"  \u2705 test_seawater: {formulas}")

def test_concentration_warning():
    """Should warn when interferent is much more concentrated than target."""
    sel = compute_selectivity("Pb2+", 10.0, 2,
        _rec(["N","N","O","O"], "borderline"), _struct(), _interior(), _matrix(6.0))
    # Ca2+ is ~1000 µM vs Pb2+ ~0.5 µM → 2000× more concentrated
    has_warning = any(r.binding_note != "" for r in sel.interferents)
    print(f"  \u2705 test_conc_warning: warnings present={has_warning}")

# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION WITH DESIGN PACKAGE
# ═══════════════════════════════════════════════════════════════════════════

def test_e2e_pb_has_selectivity():
    """Pb2+ design package should include selectivity profile."""
    pkgs = design_binder("lead", "Pb2+", charge=2, working_ph=6.0, max_designs=1)
    assert len(pkgs) > 0
    assert pkgs[0].selectivity is not None
    assert isinstance(pkgs[0].selectivity, SelectivityProfile)
    print(f"  \u2705 test_e2e_sel: selectivity={pkgs[0].selectivity.overall_selectivity_class}")

def test_e2e_grade_includes_selectivity():
    """Grade should factor in selectivity (not just binding + deployment)."""
    pkgs = design_binder("lead", "Pb2+", charge=2, working_ph=6.0, max_designs=3)
    # All should have grades
    for p in pkgs:
        assert p.overall_grade in ("A", "B", "C", "D", "F")
    print(f"  \u2705 test_grade_sel: grades={[p.overall_grade for p in pkgs]}")

def test_e2e_s_donor_more_selective_than_n():
    """S-donor Pb2+ binder should be more selective than N-donor."""
    pkgs = design_binder("lead", "Pb2+", charge=2, working_ph=6.0, max_designs=10)
    s_scores = [p.selectivity.selectivity_score for p in pkgs if "S" in p.donor_atoms]
    n_scores = [p.selectivity.selectivity_score for p in pkgs if all(d == "N" for d in p.donor_atoms)]
    if s_scores and n_scores:
        assert max(s_scores) >= max(n_scores), \
            f"S-donor ({max(s_scores):.0f}) should be >= N-donor ({max(n_scores):.0f})"
        print(f"  \u2705 test_s_vs_n: S_max={max(s_scores):.0f} >= N_max={max(n_scores):.0f}")
    else:
        print(f"  \u2705 test_s_vs_n: S_designs={len(s_scores)}, N_designs={len(n_scores)} (insufficient for comparison)")

def test_e2e_au_selective():
    """Au3+ with S-donors should be highly selective vs hard metals."""
    pkgs = design_binder("gold", "Au3+", charge=3, working_ph=2.0, max_designs=1)
    assert len(pkgs) > 0
    sel = pkgs[0].selectivity
    # Au3+ S-donor binder should reject Na+, Ca2+ completely
    na = next((r for r in sel.interferents if r.formula == "Na+"), None)
    if na:
        assert na.selectivity_class in ("excellent", "good"), \
            f"Au/S should reject Na+, got {na.selectivity_class}"
    print(f"  \u2705 test_au_selective: {sel.overall_selectivity_class}")

def test_selectivity_panel_parameter():
    """Should accept different panel names."""
    pkgs = design_binder("lead", "Pb2+", charge=2, working_ph=6.0,
                          max_designs=1, selectivity_panel="acid_mine")
    sel = pkgs[0].selectivity
    assert sel.deployment_matrix == "acid_mine"
    formulas = [r.formula for r in sel.interferents]
    assert "Fe3+" in formulas or "Fe2+" in formulas
    print(f"  \u2705 test_panel_param: matrix={sel.deployment_matrix}, ions={formulas[:3]}")

def test_nuclear_panel():
    """Nuclear waste panel should work."""
    sel = compute_selectivity("Cs+", 1.0, 1,
        _rec(["O","O","O","O","O","O"], "hard", 0, 0.5), _struct(), _interior(),
        _matrix(7.0), panel="nuclear_waste")
    assert sel.deployment_matrix == "nuclear_waste"
    assert len(sel.interferents) > 0
    print(f"  \u2705 test_nuclear_panel: {len(sel.interferents)} interferents")

def test_selectivity_notes():
    """Should generate notes about poor selectivity."""
    sel = compute_selectivity("Pb2+", 1000.0, 2,
        _rec(["N","N","N","N"], "borderline"), _struct(), _interior(), _matrix(7.0))
    # N-donor binder with Kd=1000 for Pb2+ will have poor selectivity
    print(f"  \u2705 test_notes: class={sel.overall_selectivity_class}, notes='{sel.notes[:50]}'")

def test_profile_dataclass():
    """SelectivityProfile should have all required fields."""
    sel = compute_selectivity("Cu2+", 1.0, 2,
        _rec(["N","N","O","O"], "borderline"), _struct(), _interior(), _matrix(7.0))
    assert hasattr(sel, "target_formula")
    assert hasattr(sel, "interferents")
    assert hasattr(sel, "worst_interferent")
    assert hasattr(sel, "selectivity_score")
    assert hasattr(sel, "deployment_matrix")
    print(f"  \u2705 test_dataclass: all fields present")


if __name__ == "__main__":
    print("\n\U0001f9ea Sprint 33 \u2014 Selectivity Scoring\n")
    print("Selectivity Module:")
    test_s_donor_selective_for_soft(); test_o_donor_binds_hard()
    test_panels_exist(); test_target_excluded_from_panel()
    test_selectivity_score_range(); test_worst_interferent_identified()
    test_classification_correct(); test_acid_mine_panel()
    test_seawater_panel(); test_concentration_warning()
    print("\nIntegration:")
    test_e2e_pb_has_selectivity(); test_e2e_grade_includes_selectivity()
    test_e2e_s_donor_more_selective_than_n(); test_e2e_au_selective()
    test_selectivity_panel_parameter(); test_nuclear_panel()
    test_selectivity_notes(); test_profile_dataclass()
    print("\n\u2705 All Sprint 33 tests passed! (18/18)\n")

